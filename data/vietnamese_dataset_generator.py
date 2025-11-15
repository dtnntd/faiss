"""
Vietnamese Text Dataset Generator
Tạo dataset văn bản tiếng Việt cho demo FAISS
"""

# Dataset về các chủ đề khác nhau
VIETNAMESE_TEXTS = [
    # Công nghệ
    "Trí tuệ nhân tạo đang thay đổi cách chúng ta làm việc và sinh hoạt hàng ngày.",
    "Học máy là một nhánh của trí tuệ nhân tạo tập trung vào việc xây dựng các hệ thống có thể học từ dữ liệu.",
    "Deep learning sử dụng mạng neural nhân tạo với nhiều lớp để xử lý thông tin phức tạp.",
    "Python là ngôn ngữ lập trình phổ biến nhất cho machine learning và data science.",
    "TensorFlow và PyTorch là hai framework deep learning được sử dụng rộng rãi nhất.",
    "Xử lý ngôn ngữ tự nhiên giúp máy tính hiểu và tạo ra ngôn ngữ con người.",
    "Computer vision cho phép máy tính nhận diện và phân tích hình ảnh và video.",
    "Cloud computing cung cấp khả năng tính toán và lưu trữ qua internet.",
    "Big data đề cập đến các tập dữ liệu lớn và phức tạp cần công nghệ đặc biệt để xử lý.",
    "Blockchain là công nghệ sổ cái phân tán được sử dụng trong cryptocurrency.",

    # Kinh doanh
    "Khởi nghiệp công nghệ đang thu hút nhiều đầu tư từ các quỹ venture capital.",
    "E-commerce đang phát triển mạnh mẽ tại Việt Nam với sự tăng trưởng hai con số.",
    "Marketing online giúp doanh nghiệp tiếp cận khách hàng hiệu quả hơn.",
    "Quản lý chuỗi cung ứng là yếu tố quan trọng trong thành công kinh doanh.",
    "Chuyển đổi số giúp doanh nghiệp tối ưu hóa quy trình và tăng năng suất.",
    "Dịch vụ khách hàng tốt là chìa khóa để giữ chân khách hàng lâu dài.",
    "Phân tích dữ liệu giúp doanh nghiệp đưa ra quyết định chiến lược chính xác.",
    "Nguồn nhân lực là tài sản quý giá nhất của mọi tổ chức.",
    "Đổi mới sáng tạo là động lực phát triển bền vững cho doanh nghiệp.",
    "Quản lý tài chính hiệu quả quyết định sự tồn tại của công ty.",

    # Giáo dục
    "Giáo dục trực tuyến ngày càng trở nên phổ biến sau đại dịch COVID-19.",
    "Học tập suốt đời là xu hướng cần thiết trong thời đại công nghệ.",
    "Kỹ năng mềm như giao tiếp và làm việc nhóm rất quan trọng trong công việc.",
    "Giáo dục STEM tập trung vào khoa học, công nghệ, kỹ thuật và toán học.",
    "Phương pháp giảng dạy tích cực giúp học sinh chủ động trong học tập.",
    "Đánh giá năng lực thay vì chỉ điểm số giúp phát triển toàn diện học sinh.",
    "Giáo dục đại học cần gắn kết chặt chẽ với nhu cầu thị trường lao động.",
    "Học ngoại ngữ mở ra nhiều cơ hội nghề nghiệp trong thời đại toàn cầu hóa.",
    "Giáo dục nghề nghiệp đóng vai trò quan trọng trong phát triển kinh tế.",
    "Công nghệ hỗ trợ giáo viên cá nhân hóa trải nghiệm học tập cho từng học sinh.",

    # Sức khỏe
    "Tập thể dục thường xuyên giúp cải thiện sức khỏe tim mạch và tinh thần.",
    "Chế độ ăn uống cân bằng là nền tảng của một lối sống khỏe mạnh.",
    "Ngủ đủ giấc giúp cơ thể phục hồi và tăng cường hệ miễn dịch.",
    "Stress kéo dài có thể gây ra nhiều vấn đề sức khỏe nghiêm trọng.",
    "Khám sức khỏe định kỳ giúp phát hiện sớm các bệnh lý tiềm ẩn.",
    "Yoga và thiền giúp giảm căng thẳng và cải thiện sự tập trung.",
    "Uống đủ nước mỗi ngày rất quan trọng cho các chức năng cơ thể.",
    "Hạn chế đường và muối giúp phòng ngừa bệnh tim mạch và tiểu đường.",
    "Vận động ngoài trời giúp tăng vitamin D và cải thiện tâm trạng.",
    "Chăm sóc sức khỏe tinh thần quan trọng không kém sức khỏe thể chất.",

    # Môi trường
    "Biến đổi khí hậu đang gây ra những tác động nghiêm trọng trên toàn cầu.",
    "Năng lượng tái tạo là giải pháp bền vững cho tương lai.",
    "Rác thải nhựa đang gây ô nhiễm nghiêm trọng đại dương và đất liền.",
    "Trồng cây xanh giúp giảm khí CO2 và cải thiện chất lượng không khí.",
    "Tái chế giúp giảm lượng rác thải và tiết kiệm tài nguyên thiên nhiên.",
    "Ô nhiễm không khí ảnh hưởng xấu đến sức khỏe cộng đồng.",
    "Bảo vệ đa dạng sinh học là trách nhiệm của mọi người.",
    "Sử dụng năng lượng hiệu quả giúp giảm chi phí và bảo vệ môi trường.",
    "Nông nghiệp hữu cơ thân thiện với môi trường và sức khỏe.",
    "Giảm tiêu thụ thịt giúp giảm phát thải khí nhà kính.",
]

def generate_vietnamese_dataset(n_samples: int = 1000) -> list:
    """
    Tạo dataset tiếng Việt bằng cách mở rộng và kết hợp các câu

    Args:
        n_samples: Số lượng câu cần tạo

    Returns:
        List các câu tiếng Việt
    """
    import random

    # Các từ nối và mở rộng
    connectors = [
        "Ngoài ra",
        "Bên cạnh đó",
        "Tuy nhiên",
        "Do đó",
        "Vì vậy",
        "Hơn nữa",
        "Đặc biệt",
        "Cụ thể",
        "Thực tế",
        "Theo nghiên cứu",
    ]

    extensions = [
        "điều này rất quan trọng trong thời đại hiện nay",
        "nhiều chuyên gia đã chỉ ra vấn đề này",
        "xu hướng này đang ngày càng gia tăng",
        "cần có sự quan tâm đặc biệt từ cộng đồng",
        "chúng ta cần hành động ngay từ bây giờ",
        "đây là thách thức lớn đối với xã hội",
        "việc này mang lại nhiều lợi ích to lớn",
        "cần có chiến lược dài hạn cho vấn đề này",
        "công nghệ đóng vai trò quan trọng trong lĩnh vực này",
        "giáo dục là chìa khóa để giải quyết vấn đề",
    ]

    dataset = []

    # Thêm tất cả câu gốc
    dataset.extend(VIETNAMESE_TEXTS)

    # Tạo thêm câu bằng cách kết hợp
    while len(dataset) < n_samples:
        # Lấy 1-3 câu random
        n_sents = random.randint(1, 3)
        selected = random.sample(VIETNAMESE_TEXTS, n_sents)

        if n_sents == 1:
            # Thêm extension
            sent = selected[0]
            if random.random() > 0.5:
                ext = random.choice(extensions)
                sent = f"{sent} {ext.capitalize()}."
            dataset.append(sent)
        else:
            # Kết hợp nhiều câu
            connector = random.choice(connectors)
            combined = f"{selected[0]} {connector}, {selected[1].lower()}"
            if n_sents == 3:
                combined += f" {random.choice(connectors).lower()}, {selected[2].lower()}"
            dataset.append(combined)

    return dataset[:n_samples]


def create_chunks(texts: list, chunk_size: int = 100, overlap: int = 20) -> list:
    """
    Chia văn bản thành các chunks với overlap

    Args:
        texts: List các câu
        chunk_size: Số ký tự mỗi chunk
        overlap: Số ký tự overlap giữa các chunks

    Returns:
        List các chunks
    """
    chunks = []

    for text in texts:
        if len(text) <= chunk_size:
            chunks.append(text)
        else:
            # Chia thành chunks với overlap
            start = 0
            while start < len(text):
                end = min(start + chunk_size, len(text))
                chunk = text[start:end]

                # Tìm điểm ngắt tự nhiên (dấu câu hoặc khoảng trắng)
                if end < len(text):
                    last_space = chunk.rfind(' ')
                    if last_space > chunk_size * 0.7:  # Chỉ cắt nếu không quá ngắn
                        chunk = chunk[:last_space]
                        end = start + last_space

                chunks.append(chunk.strip())
                start = end - overlap if end < len(text) else end

    return chunks


def save_dataset(filename: str = "vietnamese_dataset.txt"):
    """Tạo và lưu dataset"""
    import os

    print("Generating Vietnamese dataset...")
    texts = generate_vietnamese_dataset(1000)

    print(f"Creating chunks...")
    chunks = create_chunks(texts)

    # Save to file
    output_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(output_dir, filename)

    with open(filepath, 'w', encoding='utf-8') as f:
        for chunk in chunks:
            f.write(chunk + '\n')

    print(f"✓ Saved {len(chunks)} chunks to {filepath}")
    print(f"  Total texts: {len(texts)}")
    print(f"  Total chunks: {len(chunks)}")
    print(f"  Avg chunk length: {sum(len(c) for c in chunks) / len(chunks):.1f} chars")

    # Show samples
    print(f"\nSample chunks:")
    for i, chunk in enumerate(chunks[:5]):
        print(f"  [{i+1}] {chunk[:100]}...")

    return chunks


if __name__ == "__main__":
    save_dataset()
