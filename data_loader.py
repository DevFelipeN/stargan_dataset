from torch.utils import data
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from PIL import Image
import torch
import os
import glob


class CelebA(data.Dataset):
    """Dataset class for CelebA-style images without attribute file."""

    def __init__(
        self,
        image_dir,
        attr_path,
        selected_attrs,
        transform,
        mode
    ):
        """Initialize dataset using only the image directory."""

        self.image_dir = image_dir
        self.attr_path = attr_path
        self.selected_attrs = selected_attrs
        self.transform = transform
        self.mode = mode

        self.dataset = []

        self.preprocess()

        self.num_images = len(self.dataset)

    def preprocess(self):
        """Find all images directly from image_dir."""

        extensions = [
            '*.jpg',
            '*.jpeg',
            '*.png',
            '*.JPG',
            '*.JPEG',
            '*.PNG'
        ]

        image_files = []

        for extension in extensions:

            image_files.extend(
                glob.glob(
                    os.path.join(
                        self.image_dir,
                        extension
                    )
                )
            )

        image_files = sorted(
            list(set(image_files))
        )

        if len(image_files) == 0:

            raise RuntimeError(
                'Nenhuma imagem encontrada em: {}'.format(
                    self.image_dir
                )
            )

        # ----------------------------------------------------
        # Guardar somente o nome do arquivo.
        # O Solver continuará recebendo:
        #
        # image, label
        #
        # ----------------------------------------------------

        for image_path in image_files:

            filename = os.path.basename(
                image_path
            )

            self.dataset.append(
                filename
            )

        print(
            'Finished preprocessing the dataset...'
        )

        print(
            'Images found:',
            len(self.dataset)
        )

    def __getitem__(self, index):
        """Return one image and a dummy attribute vector."""

        filename = self.dataset[index]

        image_path = os.path.join(
            self.image_dir,
            filename
        )

        image = Image.open(
            image_path
        ).convert('RGB')

        image = self.transform(
            image
        )

        # ----------------------------------------------------
        # O Generator possui c_dim = 5:
        #
        # [Black_Hair,
        #  Blond_Hair,
        #  Brown_Hair,
        #  Male,
        #  Young]
        #
        # Como não temos o arquivo de atributos, retornamos
        # apenas um vetor de zeros.
        #
        # O novo test() do Solver NÃO utilizará esse vetor
        # para definir a transformação.
        # ----------------------------------------------------

        label = torch.zeros(
            len(self.selected_attrs),
            dtype=torch.float32
        )

        return image, label

    def __len__(self):
        return self.num_images


def get_loader(
    image_dir,
    attr_path,
    selected_attrs,
    crop_size=178,
    image_size=128,
    batch_size=16,
    dataset='CelebA',
    mode='train',
    num_workers=1
):
    """Build and return a data loader."""

    transform = []

    # --------------------------------------------------------
    # No modo test não existe RandomHorizontalFlip.
    # --------------------------------------------------------

    if mode == 'train':

        transform.append(
            T.RandomHorizontalFlip()
        )

    # --------------------------------------------------------
    # Mantém exatamente a transformação utilizada pelo
    # código original.
    # --------------------------------------------------------

    transform.append(
        T.CenterCrop(crop_size)
    )

    transform.append(
        T.Resize(image_size)
    )

    transform.append(
        T.ToTensor()
    )

    transform.append(
        T.Normalize(
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5)
        )
    )

    transform = T.Compose(
        transform
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    if dataset == 'CelebA':

        dataset = CelebA(
            image_dir,
            attr_path,
            selected_attrs,
            transform,
            mode
        )

    elif dataset == 'RaFD':

        dataset = ImageFolder(
            image_dir,
            transform
        )

    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------

    data_loader = data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=(mode == 'train'),
        num_workers=num_workers
    )

    return data_loader
