# fbot_vision_msgs

## Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/fbot_vision/fr/object_recognition` | [`Detection3DArray`](fbot_vision_msgs/msg/Detection3DArray.msg) | 3D object detections |
| `/fbot_vision/pt/tracking3D` | [`Detection3DArray`](fbot_vision_msgs/msg/Detection3DArray.msg) | 3D person tracking |
| `/fbot_vision/face_recognition/detections` | [`Detection3DArray`](fbot_vision_msgs/msg/Detection3DArray.msg) | 3D face recognition |
| `/fbot_vision/vlm/question_answering/query` | [`VLMQuestion`](fbot_vision_msgs/msg/VLMQuestion.msg) | VLM questions |
| `/fbot_vision/vlm/question_answering/answer` | [`VLMAnswer`](fbot_vision_msgs/msg/VLMAnswer.msg) | VLM responses |

## Services

| Service | Type | Description |
|---------|------|-------------|
| `/fbot_vision/fr/object_start` | [`std_srvs/Empty`](http://docs.ros.org/en/api/std_srvs/html/srv/Empty.html) | Start object detection |
| `/fbot_vision/fr/object_stop` | [`std_srvs/Empty`](http://docs.ros.org/en/api/std_srvs/html/srv/Empty.html) | Stop object detection |
| `/fbot_vision/pt/start` | [`std_srvs/Empty`](http://docs.ros.org/en/api/std_srvs/html/srv/Empty.html) | Start person tracking |
| `/fbot_vision/pt/stop` | [`std_srvs/Empty`](http://docs.ros.org/en/api/std_srvs/html/srv/Empty.html) | Stop person tracking |
| `/fbot_vision/vlm/question_answering/query` | [`VLMQuestionAnswering`](fbot_vision_msgs/srv/VLMQuestionAnswering.srv) | Ask VLM questions |
| `/fbot_vision/vlm/answer_history/query` | [`VLMAnswerHistory`](fbot_vision_msgs/srv/VLMAnswerHistory.srv) | Get VLM conversation history |
| `/fbot_vision/face_recognition/people_introducing` | [`PeopleIntroducing`](fbot_vision_msgs/srv/PeopleIntroducing.srv) | Register new person |
| `/fbot_vision/look_at_description` | [`LookAtDescription3D`](fbot_vision_msgs/srv/LookAtDescription3D.srv) | Look at specific 3D detection |