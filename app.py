
import os
import re
import time
import streamlit as st

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq


# ============================================================
# CONFIGURATION
# ============================================================

POSSIBLE_DB_PATHS = [
    
    "/content/drive/MyDrive/RAG/vectorstore/db_faiss"
]

# Retrieve more internally so that we have a better chance
# of getting continuation chunks from the same page.
RETRIEVAL_K = 8

# Used only to reject completely unrelated questions.
MAX_DISTANCE = 1.10

GROQ_MODEL = "llama-3.1-8b-instant"

FALLBACK_MESSAGE = (
    "I don't know based on the provided documents."
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Medical RAG Assistant",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 Medical RAG Assistant")


# ============================================================
# FIND FAISS DATABASE
# ============================================================

def find_database():

    for path in POSSIBLE_DB_PATHS:

        if os.path.exists(path):
            return path

    return None


DB_FAISS_PATH = find_database()


if DB_FAISS_PATH is None:

    st.error("FAISS database not found.")
    st.stop()


# ============================================================
# GROQ API KEY
# ============================================================

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")


if not GROQ_API_KEY:

    st.error("GROQ_API_KEY is not configured.")
    st.stop()


# ============================================================
# LOAD VECTORSTORE
# ============================================================

@st.cache_resource
def load_vectorstore():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.load_local(
        DB_FAISS_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    return db


# ============================================================
# LOAD GROQ
# ============================================================

@st.cache_resource
def load_llm():

    return ChatGroq(
        model=GROQ_MODEL,
        temperature=0,
        max_tokens=150,
        groq_api_key=GROQ_API_KEY
    )


# ============================================================
# INITIALIZE
# ============================================================

try:

    vectorstore = load_vectorstore()
    llm = load_llm()

except Exception as e:

    st.error(
        f"Unable to load the application: {e}"
    )

    st.stop()


# ============================================================
# CLEAN PDF TEXT
# ============================================================

def clean_source_text(text):
    """
    Cleans common PDF extraction problems.

    Example:

        pan-
        creas

    becomes:

        pancreas
    """

    if not text:
        return ""

    # Fix words broken at a line ending.
    text = re.sub(
        r"-\s*\n\s*",
        "",
        text
    )

    # Replace line breaks with spaces.
    text = re.sub(
        r"\s*\n\s*",
        " ",
        text
    )

    # Remove repeated spaces.
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# GET PAGE NUMBER
# ============================================================

def get_page_number(doc):

    page = doc.metadata.get(
        "page",
        "N/A"
    )

    try:

        return int(page)

    except Exception:

        # Try alternative metadata names.
        for key in [
            "page_number",
            "page_no",
            "pageno"
        ]:

            value = doc.metadata.get(
                key
            )

            try:

                return int(value)

            except Exception:
                pass

    return page


# ============================================================
# CHECK WHETHER CHUNK IS USELESS
# ============================================================

def is_useless_source(text):

    if not text:
        return True

    cleaned = clean_source_text(text)

    if not cleaned:
        return True

    lower = cleaned.lower().strip()

    # Very short chunks are usually headings.
    if len(cleaned) < 80:
        return True

    # Obvious reference / bibliography sections.
    bad_starts = [
        "resources",
        "references",
        "bibliography",
        "periodicals",
        "table of contents",
        "contents"
    ]

    for item in bad_starts:

        if lower.startswith(item):
            return True

    # Heading-only Gale Encyclopedia chunks.
    if re.fullmatch(
        r"gale encyclopedia of medicine.*",
        lower
    ):
        return True

    return False


# ============================================================
# CHECK WHETHER A SENTENCE IS PROPERLY TERMINATED
# ============================================================

def is_complete_sentence(sentence):
    """
    Returns True only if the sentence ends with a real
    sentence-ending punctuation mark, optionally followed
    by a closing quote or bracket.

    This is what catches fragments like:

        "Many genes produce proteins involved in controlling"

    which have no trailing punctuation at all because the
    retrieved chunk was cut off mid-sentence.
    """

    if not sentence:
        return False

    return bool(
        re.search(
            r'[.!?]["\')\]]*$',
            sentence.strip()
        )
    )


# ============================================================
# EXTRACT COMPLETE SENTENCES
# ============================================================

def get_complete_excerpt(
    text,
    max_chars=750
):
    """
    Returns only complete sentences.

    It NEVER intentionally cuts a sentence in the middle,
    and it NEVER keeps a trailing sentence that has no
    closing punctuation.

    If the retrieved chunk ends with:

        Many genes produce proteins involved in controlling

    that incomplete sentence is removed rather than shown
    as if it were a complete source excerpt.
    """

    text = clean_source_text(text)

    if not text:
        return ""

    # Split into sentences.
    sentences = re.split(
        r"(?<=[.!?])\s+",
        text
    )

    sentences = [
        s.strip()
        for s in sentences
        if s.strip()
    ]

    if not sentences:
        return ""

    # --------------------------------------------------------
    # IMPORTANT FIX:
    #
    # The regex above only splits AFTER sentence-ending
    # punctuation, so every sentence in the list is already
    # guaranteed to end properly -- EXCEPT the very last one.
    # If the raw chunk was cut off mid-sentence (very common
    # with fixed-size retrieval chunks), that trailing
    # fragment has no ".", "!", or "?" at all and would
    # otherwise be shown as if it were a finished sentence.
    #
    # Drop it before doing anything else.
    # --------------------------------------------------------

    if not is_complete_sentence(sentences[-1]):

        sentences = sentences[:-1]

    if not sentences:
        return ""

    selected = []
    current_length = 0

    for sentence in sentences:

        # If a chunk begins in the middle of a sentence,
        # skip that fragment.
        if (
            not selected
            and sentence
            and sentence[0].islower()
        ):
            continue

        additional_length = len(sentence)

        if selected:
            additional_length += 1

        if (
            current_length
            + additional_length
            <= max_chars
        ):

            selected.append(sentence)

            current_length += additional_length

        else:

            break

    # --------------------------------------------------------
    # SAFETY NET:
    #
    # Even though every sentence we kept already ends with
    # proper punctuation, double check the final selected
    # sentence one more time before returning. This guards
    # against any future edits to the loop above accidentally
    # re-introducing a trailing fragment.
    # --------------------------------------------------------

    while selected and not is_complete_sentence(selected[-1]):

        selected.pop()

    return " ".join(
        selected
    ).strip()


# ============================================================
# GET SOURCE TEXT BY PAGE
# ============================================================

def build_page_sources(
    retrieved_documents
):
    """
    Groups retrieved chunks belonging to the same page.

    This helps when FAISS retrieves two adjacent chunks
    from the same page. Their text can then be combined
    before creating the source excerpt.
    """

    page_groups = {}

    for doc, distance in retrieved_documents:

        page = get_page_number(doc)

        text = clean_source_text(
            doc.page_content
        )

        if is_useless_source(text):
            continue

        if page not in page_groups:

            page_groups[page] = []

        page_groups[page].append(
            {
                "doc": doc,
                "distance": float(distance),
                "text": text
            }
        )

    sources = []

    for page, chunks in page_groups.items():

        # Best matching chunk first.
        chunks = sorted(
            chunks,
            key=lambda x: x["distance"]
        )

        # Combine unique chunk texts.
        combined_parts = []

        seen_text = set()

        for chunk in chunks:

            text = chunk["text"]

            if text in seen_text:
                continue

            seen_text.add(text)

            combined_parts.append(text)

        combined_text = " ".join(
            combined_parts
        )

        excerpt = get_complete_excerpt(
            combined_text,
            max_chars=750
        )

        if not excerpt:
            continue

        best_distance = min(
            x["distance"]
            for x in chunks
        )

        sources.append(
            {
                "page": page,
                "text": excerpt,
                "distance": best_distance
            }
        )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # USER-FACING SOURCES ARE SORTED BY PAGE NUMBER.
    #
    # Example:
    #
    # Page 434
    # Page 435
    # Page 436
    # --------------------------------------------------------

    def page_sort_key(source):

        page = source["page"]

        try:
            return int(page)

        except Exception:
            return 999999

    sources = sorted(
        sources,
        key=page_sort_key
    )

    # Remove duplicate pages.
    final_sources = []

    seen_pages = set()

    for source in sources:

        page = source["page"]

        if page in seen_pages:
            continue

        seen_pages.add(page)

        final_sources.append(
            source
        )

    return final_sources


# ============================================================
# BUILD RAG CONTEXT
# ============================================================

def build_rag_context(
    relevant_documents
):
    """
    Keeps retrieval relevance order for the LLM.

    NOTE:
    The LLM receives relevance order.
    The USER sees page-number order.
    """

    context_parts = []

    for i, (
        doc,
        distance
    ) in enumerate(
        relevant_documents,
        start=1
    ):

        page = get_page_number(
            doc
        )

        text = clean_source_text(
            doc.page_content
        )

        context_parts.append(
            f"""
SOURCE {i}
PAGE: {page}

{text[:1800]}
"""
        )

    return "\n".join(
        context_parts
    )


# ============================================================
# RAG FUNCTION
# ============================================================

def ask_question(question):

    # --------------------------------------------------------
    # RETRIEVE MORE CHUNKS INTERNALLY
    # --------------------------------------------------------

    try:

        results = (
            vectorstore
            .similarity_search_with_score(
                question,
                k=RETRIEVAL_K
            )
        )

    except Exception as e:

        print(
            f"FAISS retrieval error: {e}"
        )

        return (
            FALLBACK_MESSAGE,
            []
        )


    if not results:

        return (
            FALLBACK_MESSAGE,
            []
        )


    # --------------------------------------------------------
    # SORT BY FAISS RELEVANCE
    #
    # Lower distance = more relevant.
    # --------------------------------------------------------

    results = sorted(
        results,
        key=lambda x: float(x[1])
    )


    best_distance = float(
        results[0][1]
    )


    # --------------------------------------------------------
    # RELEVANCE CHECK
    # --------------------------------------------------------

    if best_distance > MAX_DISTANCE:

        return (
            FALLBACK_MESSAGE,
            []
        )


    # --------------------------------------------------------
    # KEEP RELEVANT RESULTS
    # --------------------------------------------------------

    relevant_docs = []

    for doc, distance in results:

        distance = float(
            distance
        )

        if distance > MAX_DISTANCE:
            continue

        text = clean_source_text(
            doc.page_content
        )

        if is_useless_source(text):
            continue

        relevant_docs.append(
            (
                doc,
                distance
            )
        )


    if not relevant_docs:

        return (
            FALLBACK_MESSAGE,
            []
        )


    # --------------------------------------------------------
    # BUILD CONTEXT
    # --------------------------------------------------------

    context = build_rag_context(
        relevant_docs
    )


    # --------------------------------------------------------
    # STRICT RAG PROMPT
    # --------------------------------------------------------

    prompt = f"""
You are a strict document-based question answering assistant.

Answer the user's question using ONLY the information
contained in the CONTEXT.

Rules:

1. Use only the provided context.
2. Do not use outside knowledge.
3. Do not guess.
4. Do not hallucinate.
5. Do not add information that is not present in the context.
6. If the context does not contain enough information
   to answer the question, output exactly:

NOT_FOUND

7. Keep the answer clear and concise.
8. Do not mention source numbers in the answer.
9. Answer directly instead of discussing the retrieval process.

CONTEXT:

{context}

QUESTION:

{question}

ANSWER:
"""


    # --------------------------------------------------------
    # CALL GROQ
    # --------------------------------------------------------

    answer = None

    for attempt in range(2):

        try:

            response = llm.invoke(
                prompt
            )

            answer = response.content.strip()

            break

        except Exception as e:

            print(
                f"Groq attempt {attempt + 1} failed: {e}"
            )

            if attempt == 0:

                time.sleep(1)


    # --------------------------------------------------------
    # GROQ FAILURE
    # --------------------------------------------------------

    if not answer:

        return (
            "Unable to generate an answer right now. "
            "Please try again.",
            []
        )


    # --------------------------------------------------------
    # CHECK NOT_FOUND
    # --------------------------------------------------------

    normalized = (
        answer
        .strip()
        .upper()
        .replace(".", "")
    )


    if (
        normalized == "NOT_FOUND"
        or "NOT_FOUND" in normalized
    ):

        return (
            FALLBACK_MESSAGE,
            []
        )


    # --------------------------------------------------------
    # REMOVE UNNECESSARY PREFIX
    # --------------------------------------------------------

    prefixes = [
        "ANSWER:",
        "Answer:",
        "ASSISTANT:",
        "Assistant:"
    ]

    for prefix in prefixes:

        if answer.startswith(prefix):

            answer = answer[
                len(prefix):
            ].strip()


    if not answer:

        return (
            FALLBACK_MESSAGE,
            []
        )


    # --------------------------------------------------------
    # CREATE USER-FACING SOURCES
    #
    # IMPORTANT:
    #
    # Here we group same-page chunks and sort by PAGE NUMBER.
    # --------------------------------------------------------

    sources = build_page_sources(
        relevant_docs
    )


    return (
        answer,
        sources
    )


# ============================================================
# CHAT HISTORY
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


# ============================================================
# DISPLAY PREVIOUS CHAT
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


        # ----------------------------------------------------
        # PREVIOUS SOURCES
        # ----------------------------------------------------

        if (
            message["role"] == "assistant"
            and message.get("sources")
        ):

            with st.expander(
                "📚 Sources"
            ):

                for i, source in enumerate(
                    message["sources"],
                    start=1
                ):

                    st.markdown(
                        f"**Source {i} — Page {source['page']}**"
                    )

                    st.write(
                        source["text"]
                    )

                    if (
                        i
                        < len(
                            message["sources"]
                        )
                    ):

                        st.divider()


# ============================================================
# CHAT INPUT
# ============================================================

question = st.chat_input(
    "Ask a question..."
)


if question:

    # --------------------------------------------------------
    # SHOW USER QUESTION
    # --------------------------------------------------------

    with st.chat_message(
        "user"
    ):

        st.markdown(
            question
        )


    # --------------------------------------------------------
    # SAVE USER MESSAGE
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )


    # --------------------------------------------------------
    # GENERATE ANSWER
    # --------------------------------------------------------

    with st.chat_message(
        "assistant"
    ):

        with st.spinner(
            "Searching documents..."
        ):

            answer, sources = ask_question(
                question
            )


        # ----------------------------------------------------
        # SHOW ANSWER
        # ----------------------------------------------------

        st.markdown(
            answer
        )


        # ----------------------------------------------------
        # SHOW SOURCES
        # ----------------------------------------------------

        if sources:

            with st.expander(
                "📚 Sources"
            ):

                for i, source in enumerate(
                    sources,
                    start=1
                ):

                    st.markdown(
                        f"**Source {i} — Page {source['page']}**"
                    )

                    st.write(
                        source["text"]
                    )

                    if (
                        i
                        < len(sources)
                    ):

                        st.divider()


    # --------------------------------------------------------
    # SAVE ASSISTANT MESSAGE
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources
        }
    )
