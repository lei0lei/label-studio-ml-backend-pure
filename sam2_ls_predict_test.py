import json
import requests


def main():
    url = "http://127.0.0.1:9090/model/segment/sam2/sam2.1_hiera_tiny/predict"

    payload = {
        "tasks": [
            {
                "id": 1,
                "data": {
                    "image": "https://picsum.photos/640/480"
                }
            }
        ],
        "label_config": """
        <View>
          <Image name=\"image\" value=\"$image\"/>
          <PolygonLabels name=\"label\" toName=\"image\">
            <Label value=\"object\"/>
          </PolygonLabels>
          <RectangleLabels name=\"bbox\" toName=\"image\">
            <Label value=\"object\"/>
          </RectangleLabels>
        </View>
        """,
        "params": {
            "context": {
                "result": [
                    {
                        "id": "prompt-box",
                        "type": "rectanglelabels",
                        "from_name": "bbox",
                        "to_name": "image",
                        "original_width": 640,
                        "original_height": 480,
                        "value": {
                            "x": 20,
                            "y": 20,
                            "width": 60,
                            "height": 60,
                            "rotation": 0,
                            "rectanglelabels": ["object"]
                        }
                    }
                ]
            }
        }
    }

    resp = requests.post(url, json=payload, timeout=120)
    print("status:", resp.status_code)
    print(json.dumps(resp.json(), ensure_ascii=False, indent=2)[:3000])


if __name__ == "__main__":
    main()
