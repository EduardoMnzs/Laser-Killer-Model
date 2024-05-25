from PIL import Image
import os
import random
import threading

def create_image(color, index, width=64, height=64, path='images'):
    os.makedirs(path, exist_ok=True)
    img = Image.new('RGB', (width, height), color=color)
    img.save(os.path.join(path, f'{index}.png'))

# Função para gerar variações de verde
def generate_green_variations(base_green, variation_range, count, folder_name):
    threads = []
    for i in range(count):
        variation = (
            min(max(base_green[0] + random.randint(-variation_range, variation_range), 0), 255),
            min(max(base_green[1] + random.randint(-variation_range, variation_range), 0), 255),
            min(max(base_green[2] + random.randint(-variation_range, variation_range), 0), 255)
        )
        thread = threading.Thread(target=create_image, args=(variation, i, 64, 64, f'images/{folder_name}'))
        threads.append(thread)
        thread.start()
    
    for thread in threads:
        thread.join()

# Verde claro e verde escuro base
verde_claro_base = (144, 238, 144)
verde_escuro_base = (0, 100, 0)

# Geração de 100 variações de verde claro e verde escuro
generate_green_variations(verde_claro_base, 30, 500, 'verde_claro')
generate_green_variations(verde_escuro_base, 30, 500, 'verde_escuro')