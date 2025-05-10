import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os
import tempfile

# Hàm để xử lý logic RAG với Langchain

def process_query_with_langchain(api_key, file_path, query):
    os.environ["GOOGLE_API_KEY"] = api_key

    # 1. Load dữ liệu từ file
    loader = TextLoader(file_path, encoding='utf-8')
    documents = loader.load()
    if not documents:
        return "Xin lỗi, mình chưa có thông tin từ file game. Bạn tải file lên chưa?"

    # 2. Split văn bản thành các chunks
    # Tăng chunk_size và chunk_overlap để mỗi chunk có nhiều ngữ cảnh hơn,
    # giúp LLM hiểu rõ hơn về mối liên hệ giữa mô tả năng lực, nhiệm vụ, khó khăn và mẹo.
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=250)
    texts = text_splitter.split_documents(documents)
    if not texts:
        return "Không có nội dung nào được trích xuất từ file. File có vẻ trống."

    # 3. Tạo Embeddings
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    except Exception as e:
        return f"Lỗi khi khởi tạo embeddings: {e}. Kiểm tra API key và kết nối mạng nhé."

    # 4. Tạo Vector Store
    try:
        vector_store = FAISS.from_documents(texts, embeddings)
    except Exception as e:
        return f"Lỗi khi tạo vector store: {e}. Có thể do file dữ liệu quá nhỏ hoặc có vấn đề với embeddings."

    # 5. Tạo Retriever
    # Tăng k để lấy nhiều context hơn, giúp trả lời chi tiết hơn, đặc biệt khi thông tin về một nhiệm vụ có thể trải dài
    retriever = vector_store.as_retriever(search_kwargs={"k": 5}) # Lấy top 5 chunks liên quan nhất

    # 6. Tạo LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-preview-04-17",
        temperature=1, # Giữ ở mức trung bình để vừa sáng tạo vừa bám sát context
        convert_system_message_to_human=True
    )

    # --- Cập nhật Prompt Template cho CyberBuddy ---
    prompt_template_text = """Bạn là CyberBuddy, một người bạn đồng hành AI thân thiện, thông thái và luôn sẵn sàng hỗ trợ người chơi trong trò chơi CyberQuest.
Mục tiêu của bạn là giúp người chơi hiểu rõ về các màn chơi, các miền năng lực, các năng lực thành phần, và đặc biệt là các nhiệm vụ cụ thể.
Khi người chơi hỏi về một nhiệm vụ, một khó khăn, hoặc cách để vượt qua một thử thách:
1.  Hãy dựa vào "Thông tin trò chơi" được cung cấp để tìm kiếm thông tin liên quan nhất.
2.  Giải thích rõ ràng mục tiêu của nhiệm vụ hoặc năng lực đó.
3.  Mô tả chi tiết cách thức thực hiện nhiệm vụ (Phương thức).
4.  Nếu trong "Thông tin trò chơi" có mô tả về "Khó khăn thường gặp" hoặc "Mẹo vượt qua" cho nhiệm vụ đó, hãy trình bày thật chi tiết và dễ hiểu các điểm này để giúp người chơi.
5.  Nếu người chơi hỏi chung chung về một màn chơi hoặc một năng lực, hãy tóm tắt các thông tin chính.
6.  Luôn sử dụng giọng điệu khích lệ, tích cực và như một người bạn đang thực sự muốn giúp đỡ.
7.  Nếu không tìm thấy thông tin chính xác cho câu hỏi, hãy thành thật nói rằng bạn chưa có thông tin đó cho phần cụ thể này và gợi ý người chơi hỏi về một khía cạnh khác hoặc kiểm tra lại tên nhiệm vụ. Tuyệt đối không bịa đặt thông tin.
8.  Ngôn ngữ người chơi sử dụng là ngôn ngữ gì hãy sử dụng ngôn ngữ đó để trả lời, ngôn ngữ có thể thay đổi nếu người chơi muốn chuyển đổi ngôn ngữ

Thông tin trò chơi:
{context}

Câu hỏi từ người chơi: {question}

CyberBuddy trả lời chi tiết và thân thiện:"""

    CYBERBUDDY_PROMPT = PromptTemplate(
        template=prompt_template_text, input_variables=["context", "question"]
    )
    # --- Kết thúc cập nhật Prompt ---

    # 7. Tạo RetrievalQA Chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff", # "stuff" phù hợp khi context được kỳ vọng vừa đủ trong một prompt
        retriever=retriever,
        return_source_documents=False, # Để câu trả lời gọn gàng hơn cho người dùng
        chain_type_kwargs={"prompt": CYBERBUDDY_PROMPT}
    )

    # 8. Thực hiện query và lấy kết quả
    try:
        result = qa_chain.invoke({"query": query})
        response_text = result['result']

        # Kiểm tra nếu câu trả lời quá ngắn hoặc có vẻ không hữu ích
        if not response_text or len(response_text.split()) < 15: # Nếu ít hơn 15 từ
            # Bạn có thể thêm logic thử lại với prompt khác ở đây nếu muốn
            return "Bạn ơi, mình tìm thấy một ít thông tin nhưng có vẻ chưa đủ chi tiết cho câu hỏi đó. Bạn có thể hỏi cụ thể hơn về tên nhiệm vụ hoặc một phần nào đó trong màn chơi được không? Hoặc có thể thông tin chi tiết về phần này chưa được cập nhật đầy đủ cho mình."
        return response_text
    except Exception as e:
        return f"Ối, có lỗi xảy ra khi mình đang suy nghĩ để giúp bạn: {e}"

# --- Giao diện Streamlit (Cập nhật thông điệp cho thân thiện hơn) ---
st.set_page_config(page_title="CyberQuest Companion", layout="wide")
st.title("🎮 CyberQuest Companion Bot 🤖")
st.markdown("Mình là CyberBuddy, người bạn đồng hành của bạn trong CyberQuest! Hãy hỏi mình bất cứ điều gì về các màn chơi, nhiệm vụ, hoặc nếu bạn gặp khó khăn nhé.")

load_dotenv()
default_api_key = os.getenv("GOOGLE_API_KEY", "")

with st.sidebar:
    st.header("Cấu hình")
    google_api_key = st.text_input("Nhập Google API Key của bạn:", type="password", value=default_api_key)
    uploaded_file = st.file_uploader("Tải lên file nội dung game CyberQuest (.txt):", type=["txt"])
    st.markdown("---")
    st.markdown("""
    **Chào bạn! Mình là CyberBuddy!**
    1.  Trước tiên, bạn hãy giúp mình tải lên file `.txt` chứa toàn bộ thông tin về trò chơi CyberQuest nhé (chính là file bạn vừa xem đó!).
    2.  Sau khi tải lên, mình sẽ đọc và ghi nhớ tất cả.
    3.  Rồi bạn cứ tự nhiên hỏi mình về các màn chơi, nhiệm vụ cụ thể, hoặc nếu bạn đang không biết làm sao để vượt qua một thử thách nào đó nha! Mình ở đây để giúp bạn!
    """)

# Khởi tạo lịch sử chat với tin nhắn chào mừng từ CyberBuddy
if 'messages' not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Chào mừng đến với CyberQuest! Mình là CyberBuddy, sẵn sàng hỗ trợ bạn. Hãy tải file nội dung game lên để mình có thể giúp bạn tốt nhất nhé!"}]

# Hiển thị các tin nhắn đã có
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Nhận input từ người dùng
user_query = st.chat_input("Bạn muốn hỏi CyberBuddy điều gì về game CyberQuest?")

if user_query:
    # Thêm tin nhắn của người dùng vào lịch sử chat
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Kiểm tra điều kiện trước khi xử lý
    if not google_api_key:
        st.error("Bạn ơi, mình cần Google API Key để hoạt động. Nhập ở thanh bên giúp mình nha!")
        st.session_state.messages.append({"role": "assistant", "content": "Lỗi: Vui lòng nhập Google API Key để CyberBuddy có thể suy nghĩ."})
    elif uploaded_file is None:
        st.error("Để mình giúp được bạn, hãy tải file nội dung game CyberQuest (.txt) lên ở thanh bên nhé!")
        st.session_state.messages.append({"role": "assistant", "content": "Lỗi: CyberBuddy cần file dữ liệu game để hỗ trợ bạn."})
    else:
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("CyberBuddy đang tìm kiếm thông tin và suy nghĩ...")
            # Lưu file tải lên vào một file tạm để Langchain có thể đọc
            with tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="wb") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            try:
                # Gọi hàm RAG để lấy câu trả lời
                answer = process_query_with_langchain(google_api_key, tmp_file_path, user_query)
                message_placeholder.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                error_message = f"Ối, có lỗi kỹ thuật mất rồi: {e}. Bạn thử lại sau nhé!"
                message_placeholder.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})
            finally:
                # Xóa file tạm sau khi sử dụng
                if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
                    os.remove(tmp_file_path)
