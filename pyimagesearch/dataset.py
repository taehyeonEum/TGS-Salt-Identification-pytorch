from torch.utils.data import Dataset 
import cv2

class SegmentationDataset(Dataset):
    def __init__(self, imagePathList, maskPathList, transforms):

        self.imageList = imagePathList#list of input image paths
        self.maskList = maskPathList  #path of ground truth
        self.transforms = transforms

    def __len__(self):
        return len(self.imageList)

    def __getitem__(self, idx):
        selected_image = self.imageList[idx]

        image = cv2.imread(selected_image) # what is type of selected_image???
        image = cv2.cvtColor(image,  cv2.COLOR_BGR2RGB) #change color sequence BGR -> RGB
        mask = cv2.imread(self.maskList[idx], 0) #flag 0 is 'grayscale'

        if self.transforms is not None:
            image = self.transforms(image)
            mask = self.transforms(mask)

        return (image, mask) # getitem returns transformed image and mask!