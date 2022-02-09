import supervisely as sly

def main():
    api = sly.Api.from_env()

    # task id of the deployed model
    task_id = 13227

    # get model info
    response = api.task.send_request(task_id, "get_session_info", data={}, timeout=1)
    print("APP returns data:")
    print(response)

    # get masks for image by url
    predictions = api.task.send_request(task_id, "inference_image_url", data={
        'image_url': 'https://img.icons8.com/color/1000/000000/deciduous-tree.png'
    }, timeout=60)
    print("APP returns data:")
    print(predictions)


if __name__ == "__main__":
    main()