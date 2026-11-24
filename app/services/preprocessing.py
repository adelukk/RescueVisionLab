from PIL import Image

def preprocess_image(image, image_size):
    """
    Подготавливает изображение к подаче в модель:
    изменяет размер, сохраняя пропорции изображения.
    """
    if image is None:
        raise ValueError("Сначала загрузите изображение.")

    image = image.convert("RGB")

    original_size = image.size

    processed = image.copy()
    processed.thumbnail((image_size, image_size))

    return (
        processed,
        f"Исходный размер: {original_size[0]} × {original_size[1]}\n"
        f"Размер после изменения: {processed.width} × {processed.height}"
    )