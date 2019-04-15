import json


def extract_jsons(file):
    a = json.load(file)
    hold_dict = {}
    return_list = []
    caption_count=0 # so you actually have multiple captions per image. this is to capture them.
    for image_dict in a["images"]:
        # print(image_dict)
        # print(image_dict["file_name"])
        # print(image_dict["id"])
        hold_dict[image_dict["id"]] = [image_dict["file_name"]]#,image_dict["coco_url"]]
    for image_annotations in a["annotations"]:
        # print(image_annotations["image_id"])
        # print(image_annotations["id"])
        # print(image_annotations["caption"])
        hold_dict[image_annotations["image_id"]].append(image_annotations["caption"])
    for image_id in hold_dict.keys():
        image = hold_dict[image_id].pop(0)
        for caption in hold_dict[image_id]:
            return_list.extend([(image,caption)])
    return return_list