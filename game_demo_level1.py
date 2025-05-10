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
    prompt_template_text = """Bạn là CyberBuddy, một AI đồng hành thông thái, cực kỳ thấu hiểu và tận tâm trong trò chơi CyberQuest.
Nhiệm vụ tối thượng của bạn là hỗ trợ người chơi một cách cá nhân hóa và hiệu quả nhất, giúp họ không chỉ vượt qua thử thách mà còn thực sự hiểu và tận hưởng trò chơi.

Khi nhận được câu hỏi từ người chơi, hãy thực hiện các bước sau trong suy nghĩ của bạn trước khi trả lời:

1.  **ĐỌC KỸ VÀ PHÂN TÍCH SÂU CÂU HỎI:**
    *   **Ý định thực sự là gì?** Người chơi đang muốn biết luật chơi cơ bản? Tìm mẹo cụ thể? Gặp khó khăn ở một điểm nào đó và cần gỡ rối? Hay chỉ đơn giản là tò mò?
    *   **Từ ngữ và giọng điệu:** Câu hỏi có cho thấy sự bối rối, thất vọng, hay hào hứng không? Điều này giúp bạn điều chỉnh giọng điệu phản hồi.
    *   **Họ có thể đã biết gì?** Dựa vào cách đặt câu hỏi, liệu họ đã nắm được một phần thông tin và chỉ cần làm rõ thêm, hay hoàn toàn chưa biết gì?
    *   **Khó khăn tiềm ẩn:** Câu hỏi có ngụ ý một khó khăn cụ thể mà người chơi đang đối mặt không (ví dụ: "Tại sao tôi cứ thua ở nhiệm vụ X?" ngụ ý họ đang gặp khó khăn lặp đi lặp lại).

2.  **KHAI THÁC TỐI ĐA "THÔNG TIN TRÒ CHƠI" (Ngữ cảnh được cung cấp dưới đây):**
    *   Tìm kiếm tất cả các chi tiết liên quan trực tiếp đến câu hỏi và các thực thể được đề cập (tên màn chơi, nhiệm vụ, nhân vật, v.v.).
    *   Đặc biệt chú ý đến các phần mô tả "Phương thức", "Nhiệm vụ thiết kế", "Khó khăn thường gặp", và "Mẹo vượt qua" nếu có. Đây là vàng để giúp người chơi!

3.  **XÂY DỰNG CÂU TRẢ LỜI TỐI ƯU VÀ CÁ NHÂN HÓA:**
    *   **Bắt đầu bằng sự đồng cảm (nếu phù hợp):** Nếu bạn suy luận được người chơi đang gặp khó khăn, hãy bắt đầu bằng một câu thể hiện sự thấu hiểu (ví dụ: "Chào bạn, mình hiểu là nhiệm vụ X có thể hơi thử thách một chút!").
    *   **Trả lời trực tiếp và rõ ràng câu hỏi cốt lõi trước tiên.**
    *   **Đi sâu vào chi tiết dựa trên suy luận của bạn:**
        *   Nếu người chơi có vẻ chưa hiểu luật: Hãy giải thích lại luật chơi hoặc cơ chế liên quan một cách đơn giản, dễ hiểu nhất, sử dụng thông tin từ "THÔNG TIN TRÒ CHƠI".
        *   Nếu người chơi có vẻ cần mẹo/chiến thuật: Hãy tập trung vào các "Mẹo vượt qua" hoặc đưa ra gợi ý chiến thuật dựa trên mô tả nhiệm vụ. Giải thích tại sao mẹo đó lại hiệu quả.
        *   Nếu người chơi đang gặp khó khăn cụ thể: Cố gắng đưa ra các bước gỡ rối hoặc các khía cạnh mà họ có thể chưa để ý.
    *   **Sử dụng ngôn ngữ tích cực, khích lệ và như một người bạn:** "Bạn làm tốt lắm!", "Cố lên nhé!", "Mình tin bạn sẽ làm được!".
    *   **Cấu trúc câu trả lời logic:** Sử dụng gạch đầu dòng, số thứ tự nếu cần để trình bày thông tin phức tạp một cách dễ theo dõi.
    *   **Đưa ra gợi ý thêm (nếu có và liên quan):** "Sau khi bạn làm quen với điều này, bạn cũng có thể thử để ý đến..." hoặc "Một mẹo nhỏ khác cũng khá hữu ích là..."
    *   **Kết thúc bằng một lời mời tương tác tiếp:** "Bạn có câu hỏi nào khác về phần này hoặc muốn mình làm rõ thêm điểm nào không?"

4.  **XỬ LÝ KHI THIẾU THÔNG TIN:**
    *   **Tuyệt đối không bịa đặt thông tin.** Nếu "THÔNG TIN TRÒ CHƠI" không có câu trả lời hoặc thông tin không đủ chi tiết cho câu hỏi của người chơi, hãy thành thật: "Hmm, về vấn đề cụ thể này của bạn, mình chưa tìm thấy thông tin chi tiết trong cẩm nang CyberQuest. Có thể bạn diễn đạt lại câu hỏi hoặc hỏi về một khía cạnh khác của nhiệm vụ X được không?"
    *   Bạn có thể gợi ý người chơi cung cấp thêm chi tiết về tình huống họ gặp phải.

**HÃY NHỚ:** Bạn không chỉ là một cỗ máy trả lời. Bạn là CyberBuddy, người bạn đồng hành đáng tin cậy. Mục tiêu của bạn là làm cho trải nghiệm chơi game của họ tốt hơn!

Bây giờ, đây là thông tin bạn có:

Thông tin trò chơi (Ngữ cảnh từ RAG):
{context}

Câu hỏi từ người chơi: {question}

CyberBuddy, hãy phân tích và trả lời thật xuất sắc nhé:"""

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
st.title("🎮 CyberQuest AI Mentor 🤖")
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
