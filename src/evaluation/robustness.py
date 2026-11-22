from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter
import numpy as np


def darken_image(image, factor=0.5):
    """
    Затемнение изображения.
    factor < 1 делает изображение темнее.
    """

    enhancer = ImageEnhance.Brightness(image)

    return enhancer.enhance(factor)


def gaussian_blur(image, radius=3):
    """
    Гауссово размытие.
    """

    return image.filter(
        ImageFilter.GaussianBlur(radius)
    )


def gaussian_noise(image, sigma=20):
    """
    Добавление гауссовского шума.
    """

    array = np.asarray(image).astype(np.float32)

    noise = np.random.normal(
        0,
        sigma,
        array.shape
    )

    noisy = array + noise

    noisy = np.clip(
        noisy,
        0,
        255
    ).astype(np.uint8)

    return Image.fromarray(noisy)


def jpeg_compression(image, quality=30):
    """
    Имитация потери качества JPEG.
    """

    import io

    buffer = io.BytesIO()

    image.save(
        buffer,
        format="JPEG",
        quality=quality
    )

    buffer.seek(0)

    return Image.open(buffer).convert("RGB")


def partial_occlusion(
    image,
    fraction=0.25
):
    """
    Простая искусственная окклюзия.

    Закрывает часть изображения.
    """

    image = image.copy()

    width, height = image.size

    occlusion_width = int(
        width * fraction
    )

    occlusion_height = int(
        height * fraction
    )

    x1 = width // 2 - occlusion_width // 2
    y1 = height // 2 - occlusion_height // 2

    x2 = x1 + occlusion_width
    y2 = y1 + occlusion_height

    array = np.asarray(image).copy()

    array[y1:y2, x1:x2] = 0

    return Image.fromarray(array)


def create_variants(
    image_path,
    output_dir
):

    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    image = Image.open(
        image_path
    ).convert("RGB")

    variants = {
        "original": image,

        "dark": darken_image(
            image,
            factor=0.5
        ),

        "blur": gaussian_blur(
            image,
            radius=3
        ),

        "noise": gaussian_noise(
            image,
            sigma=20
        ),

        "jpeg": jpeg_compression(
            image,
            quality=30
        ),

        "occlusion": partial_occlusion(
            image,
            fraction=0.25
        ),
    }

    for name, variant in variants.items():

        variant.save(
            output_dir / f"{name}.jpg",
            quality=95
        )


if __name__ == "__main__":

    print(
        "Модуль robustness.py загружен."
    )