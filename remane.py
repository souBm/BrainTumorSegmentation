### replacing all file name in a folder to 1.png,2.png ... n.png , n=end image in that folder. do this using python
import glob
import os
#input_folder="C:/Users/user2/OneDrive/Desktop/FINALYR_PROJECT/PROJECT_DATASET/PROJECT_RANDOM_FOREST/DATA/p5/image"
mask_folder="./DATA/test-images"
def rename_files_in_folder(folder_path):
    
    # Get all files in the folder
    files = glob.glob(os.path.join(folder_path, '*'))
    print(files)
    # Filter out directories, we only want files
    files = [f for f in files if os.path.isfile(f)]

    for i, file_path in enumerate(files, start=0):
        # Get the file extension
        _, extension = os.path.splitext(file_path)

        # Construct new file path
        new_file_path = os.path.join(folder_path, f'{i}{extension}')

        # Rename the file
        os.rename(file_path, new_file_path)


    print(file_path)
    return files

#rename_files_in_folder(input_folder)

rename_files_in_folder(mask_folder)