import os 
import argparse

def rename_file_and_labels():
    count = 7
    print(os.environ['image_path'])

    for img in os.listdir(os.environ['image_path']):
        # os.rename(os.environ['image_path'] + img, count + 'jpg' )
        
        # print( os.path.join(os.environ['label_path'] , img.split(".")[0] + '.json'))
        if( os.path.isfile( os.path.join(os.environ['label_path'] , img.split(".")[0] + '_keypoints.json') )):
            # print( os.path.join(os.environ['image_path'], img))
            os.rename( os.path.join(os.environ['image_path'], img), os.path.join(os.environ['label_path'],str(count) + '.jpg' ))
            os.rename(os.path.join(os.environ['label_path'] , img.split(".")[0] + '_keypoints.json'), os.path.join(os.environ['label_path'],str(count) + '.json' ))
            count +=1


if __name__ == "__main__":
    print('ou')
    # initialize ArgumentParser class of argparse
    parser = argparse.ArgumentParser()
    
    # where the original images (used for creating the synthetic images) are  
    parser.add_argument(
    "--image_path",
    type=str
    )

    parser.add_argument(
    "--label_path",
    type=str
    )


    args = parser.parse_args()
    assert(args.image_path is not None)
    print(args.image_path)
    os.environ['image_path'] = args.image_path
    assert(args.label_path is not None)
    os.environ['label_path'] = args.label_path
    rename_file_and_labels()
