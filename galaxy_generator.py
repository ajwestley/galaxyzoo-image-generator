from argparse import ArgumentParser
from image_generator import ImageGenerator

DEFAULT_CONFIG_PATH = 'models/GalaxyGen.json'


def main():
    parser = ArgumentParser(
            prog='Galaxy image generator',
            description='Generate images of galaxies using a pretrained VQ-VAE.'
        )
    
    parser.add_argument('-t', '--temperature', default=1.0)
    parser.add_argument('-n', '--num_images', default=1)
    parser.add_argument('-o', '--output', default='output.png')
    parser.add_argument('-v', '--verbose', action='store_true')
    
    args = parser.parse_args()
    
    num = int(args.num_images)
    temperature = float(args.temperature)
    outfile = args.output
    
    generator = ImageGenerator(DEFAULT_CONFIG_PATH)
    
    if not (outfile.lower().endswith('.png') or outfile.lower().endswith('.jpg') or outfile.lower().endswith('.jpeg')):
        outfile = outfile + '.png'
    if num < 1:
        print(f'Cannot generate {num} images.')
        exit()
    elif num == 1:
        generator.generate_image(outfile, temperature)
    else:
        file_parts = outfile.split('.')
        base = file_parts[0] if len(file_parts) == 2 else '.'.join(file_parts[:-1])
        ext = file_parts[-1]
        for i in range(1, num+1):
            filename = f'{base}{i}.{ext}'
            generator.generate_image(filename, temperature)

if __name__ == '__main__':
    main()