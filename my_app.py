import streamlit as st
from PIL import Image
from os import write
# openpose requirement
import cv2
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
from random import randint

import pickle
import mediapipe as mp
from streamlit_webrtc import (
    AudioProcessorBase,
    ClientSettings,
    VideoTransformerBase,
    WebRtcMode,
    webrtc_streamer,
)

WEBRTC_CLIENT_SETTINGS = ClientSettings(
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
    )


icon = Image.open("images/icon.jpg")
st.set_page_config(
    page_title="REAL TIME HUMAN POSE ESTIMATION:",
    page_icon=icon,
    layout="wide",
    initial_sidebar_state="auto",
)

st.sidebar.markdown("<h1 style='color:Red ;'>Navigation</h1>", unsafe_allow_html=True)
page = st.sidebar.radio(" ",("Home","BlazePose","OpenPose"))

if page == "BlazePose":

    header = st.beta_container()
    explain = st.beta_container()
    pre = st.beta_container()

    with header:
        st.markdown("<h1 style='text-align: center; color:Red ;'>REAL TIME HUMAN POSE ESTIMATION: </h1>", unsafe_allow_html=True)
                    

    with explain:
        st.markdown("## BLAZEPOSE :")
        st.write("BlazePose, a lightweight convolutional neural network architecture for Single person pose estimation that is tailored for real-time. During inference, the network produces 33 body keypoints for a single person and runs about 30 frames per second. This makes it particularly suited to real-time use cases like fitness tracking and sign language recognition and many more.")
        blaze1 = ['images/blazepose4.gif', 'images/blazepose5.gif']
        st.image(blaze1, use_column_width=True* len(blaze1))
        st.write("The current standard for human body pose is the COCO topology, which consists of 17 landmarks across the torso, arms, legs, and face. However, the COCO keypoints only localize to the ankle and wrist points, lacking scale and orientation information for hands and feet, which is vital for practical applications like fitness and dance. The inclusion of more keypoints is crucial for the subsequent application of domain-specific pose estimation models, like those for hands, face, or feet.")
        st.markdown("### Topology: ")
        st.write("With BlazePose,a new topology of 33 human body keypoints, which is a superset of COCO, BlazeFace and BlazePalm topologies. This allows us to determine body semantics from pose prediction alone that is consistent with face and hand models.")
        _,col2,_ = st.beta_columns([2,2,2])
        with col2:
            st.image("images/blazepose_key.png")
        code = '''0. Nose, 1. Left eye inner, 2. Left eye, 3. Left eye outer, 4. Right eye inner, 5. Right eye, 6. Right eye outer, 7. Left ear, 8. Right ear, 9. Mouth left 10. Mouth right, 
    11. Left shoulder, 12. Right shoulder, 13. Left elbow, 14. Right elbow, 15. Left wrist, 16. Right wrist, 17. Left knuckle , 18. Right knuckle, 19. Left index, 20. Right index, 
    21. Left thumb , 22. Right thumb , 23. Left hip, 24. Right hip, 25. Left knee, 26. Right knee, 27. Left ankle, 28. Right ankle, 29. Left heel, 30. Right heel, 31. Left foot index, 
    32. Right foot index '''
        st.code(code)
                    
        st.markdown("<h2 style='text-align: center; color:Red ;'>Model: </h2>", unsafe_allow_html=True)
        st.markdown("### The Model predict's the Yoga posture of a single person either he/she is doing Poses like **1) Tadasan**,**2) Balancing**,**3) Warrior Pose**,**4) Padmasana** :walking: :wrestlers:")
                
    with pre:
        st.sidebar.markdown("<h1 style='color:Red ;'>References</h1>", unsafe_allow_html=True)
        st.sidebar.markdown("[On-device, Real-time Body Pose Tracking with MediaPipe BlazePose](https://ai.googleblog.com/2020/08/on-device-real-time-body-pose-tracking.html)")
        st.sidebar.markdown("[BlazePose: On-device Real-time Body Pose tracking](https://arxiv.org/abs/2006.10204)")

        st.sidebar.markdown("<h2 style='color:Red ;'>If you like the Content Press Button and Enjoy: </h2>", unsafe_allow_html=True)
        if st.sidebar.button("Press if You Like"):
            st.balloons() 

    class VideoTransformer(VideoTransformerBase):
        mp_drawing = mp.solutions.drawing_utils # Drawing helpers
        mp_holistic = mp.solutions.holistic # Mediapipe Solutions


        def load_model():
            with open('body_language.pkl', 'rb') as f:
                model = pickle.load(f)
            return model
        model = load_model()

        def blazepose():
            cap = cv2.VideoCapture(0)
                    # Initiate holistic model
            with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
                        
                while cap.isOpened():
                    ret, frame = cap.read()
                            
                            # Recolor Feed
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image.flags.writeable = False        
                            
                            # Make Detections
                    results = holistic.process(image)
                            # print(results.face_landmarks)
                            
                            # face_landmarks, pose_landmarks, left_hand_landmarks, right_hand_landmarks
                            
                            # Recolor image back to BGR for rendering
                    image.flags.writeable = True   
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                            
                            # 1. Draw face landmarks
                    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS, 
                                            mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                                            mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                                            )
                            
                            # 2. Right hand
                    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                            mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                                            mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                                            )

                            # 3. Left Hand
                    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                            mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                                            mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                                            )

                            # 4. Pose Detections
                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                            )
                            # Export coordinates
                    try:
                                # Extract Pose landmarks
                        pose = results.pose_landmarks.landmark
                        pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())
                                
                                # Extract Face landmarks
                        face = results.face_landmarks.landmark
                        face_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in face]).flatten())
                                
                                # Concate rows
                        row = pose_row+face_row
                                


                                # Make Detections
                        X = pd.DataFrame([row])
                        body_language_class = model.predict(X)[0]
                        body_language_prob = model.predict_proba(X)[0]
                        print(body_language_class, body_language_prob)
                                
                                # Grab ear coords
                        coords = tuple(np.multiply(
                                        np.array(
                                            (results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].x, 
                                            results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].y))
                                    , [640,480]).astype(int))
                                
                        cv2.rectangle(image, 
                                    (coords[0], coords[1]+5), 
                                    (coords[0]+len(body_language_class)*20, coords[1]-30), 
                                    (245, 117, 16), -1)
                        cv2.putText(image, body_language_class, coords, 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                                
                                # Get status box
                        cv2.rectangle(image, (0,0), (250, 60), (245, 117, 16), -1)
                                
                                # Display Class
                        cv2.putText(image, 'CLASS'
                                    , (95,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                        cv2.putText(image, body_language_class.split(' ')[0]
                                    , (90,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                                
                                # Display Probability
                        cv2.putText(image, 'PROB'
                                    , (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                        cv2.putText(image, str(round(body_language_prob[np.argmax(body_language_prob)],2))
                                    , (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                                
                    except:
                        pass
                                            
                            # cv2.imshow('Raw Webcam Feed', image)
                    return image

                            # if cv2.waitKey(2000) & 0xFF == ord('q'):
                            #     break

            # cap.release()
            # cv2.destroyAllWindows()


    webrtc_ctx = webrtc_streamer(key="Blazepose", mode=WebRtcMode.SENDRECV, client_settings=WEBRTC_CLIENT_SETTINGS, video_transformer_factory=VideoTransformer,async_transform=True,)
    
#=====================================================================================================
# ------------------------------------------OPenPose--------------------------------------------------
#=====================================================================================================
   
elif page == "OpenPose":
    header = st.beta_container()
    explain = st.beta_container()
    model = st.beta_container()
    pre = st.beta_container()

    with header:
        st.markdown("<h1 style='text-align: center; color:Red ;'>REAL TIME HUMAN POSE ESTIMATION: </h1>", unsafe_allow_html=True)

    with explain:
        st.markdown("## OPENPOSE :man-bouncing-ball:")
        st.write("Openpose is used for Single-Person as well as on Multi-Person for real-time human pose estimation from video/image")
        st.write("Multi-Person pose estimation is more difficult than the single person case as the location and the number of people in an image are unknown. Typically, we can tackle the above issue using one of two approaches")
        st.markdown("#### Top Down Approach:")
        st.write("In this Approach First Machine Tracks The Human Body and then Calculate the pose for each person.")
        st.markdown("#### Bottom Up Approach:")
        st.write("This Work Heavily optimizes the OpenPose and Bottom-Up Approach.First it Detects the Skeleton(which consist of keypoint and connections between them) to identify human poses and  contains up to nineteen keypoints")
        st.markdown("### Topology:")
        st.write("This Model is trained on COCO dataset(Common Objects In Context) COCO dataset model detects 17 keypoints + 1 background on the body as stated above.")
        _,col2,_ = st.beta_columns([2,2,2])
        with col2:
            st.image("images/openpose_key.png")
        code = '''0. Nose, 1. Neck, 2. RShoulder, 3. RElbow, 4. RWrist, 5. LShoulder, 6. LElbow, 7. LWrist, 8. RHip, 9. RKnee 10. RAnkle,  11. LHip, 12. LKnee, 13. LAnkle, 14. REye, 
15. LEye, 16. REar, 17.LEar , 18. Background  '''

        st.code(code)
    with model:
        st.markdown("<h2 style='text-align: center; color:Red ;'>Model: </h2>", unsafe_allow_html=True)
         ### Here i am here working on static image and model is caffemodel
        st.markdown("### For Checking out on Static image you should upload images into file uploader and it will provide output :person_doing_cartwheel:")
        protoFile = "models/coco/pose_deploy_linevec.prototxt"
        weightsFile = "models/coco/pose_iter_440000.caffemodel"
        nPoints = 18
        # COCO Output Format
        keypointsMapping = ['Nose', 'Neck', 'R-Sho', 'R-Elb', 'R-Wr', 'L-Sho', 
                            'L-Elb', 'L-Wr', 'R-Hip', 'R-Knee', 'R-Ank', 'L-Hip', 
                            'L-Knee', 'L-Ank', 'R-Eye', 'L-Eye', 'R-Ear', 'L-Ear']

        POSE_PAIRS = [[1,2], [1,5], [2,3], [3,4], [5,6], [6,7],
                    [1,8], [8,9], [9,10], [1,11], [11,12], [12,13],
                    [1,0], [0,14], [14,16], [0,15], [15,17],
                    [2,17], [5,16] ]

        # index of pafs correspoding to the POSE_PAIRS
        # e.g for POSE_PAIR(1,2), the PAFs are located at indices (31,32) of output, Similarly, (1,5) -> (39,40) and so on.
        mapIdx = [[31,32], [39,40], [33,34], [35,36], [41,42], [43,44], 
                [19,20], [21,22], [23,24], [25,26], [27,28], [29,30], 
                [47,48], [49,50], [53,54], [51,52], [55,56], 
                [37,38], [45,46]]

        colors = [ [0,100,255], [0,100,255], [0,255,255], [0,100,255], [0,255,255], [0,100,255],
                [0,255,0], [255,200,100], [255,0,255], [0,255,0], [255,200,100], [255,0,255],
                [0,0,255], [255,0,0], [200,200,0], [255,0,0], [200,200,0], [0,0,0]]


        # Find the Keypoints using Non Maximum Suppression on the Confidence Map
        def getKeypoints(probMap, threshold=0.1):
            
            mapSmooth = cv2.GaussianBlur(probMap,(3,3),0,0)

            mapMask = np.uint8(mapSmooth>threshold)
            keypoints = []
            
            #find the blobs
            contours, _ = cv2.findContours(mapMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            #contours,hierachy=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            
            #for each blob find the maxima
            for cnt in contours:
                blobMask = np.zeros(mapMask.shape)
                blobMask = cv2.fillConvexPoly(blobMask, cnt, 1)
                maskedProbMap = mapSmooth * blobMask
                _, maxVal, _, maxLoc = cv2.minMaxLoc(maskedProbMap)
                keypoints.append(maxLoc + (probMap[maxLoc[1], maxLoc[0]],))

            return keypoints

        # Find valid connections between the different joints of a all persons present
        def getValidPairs(output):
            valid_pairs = []
            invalid_pairs = []
            n_interp_samples = 10
            paf_score_th = 0.1
            conf_th = 0.7
            # loop for every POSE_PAIR
            for k in range(len(mapIdx)):
                # A->B constitute a limb
                pafA = output[0, mapIdx[k][0], :, :]
                pafB = output[0, mapIdx[k][1], :, :]
                pafA = cv2.resize(pafA, (frameWidth, frameHeight))
                pafB = cv2.resize(pafB, (frameWidth, frameHeight))

                # Find the keypoints for the first and second limb
                candA = detected_keypoints[POSE_PAIRS[k][0]]
                candB = detected_keypoints[POSE_PAIRS[k][1]]
                nA = len(candA)
                nB = len(candB)

                # If keypoints for the joint-pair is detected
                # check every joint in candA with every joint in candB 
                # Calculate the distance vector between the two joints
                # Find the PAF values at a set of interpolated points between the joints
                # Use the above formula to compute a score to mark the connection valid
                
                if( nA != 0 and nB != 0):
                    valid_pair = np.zeros((0,3))
                    for i in range(nA):
                        max_j=-1
                        maxScore = -1
                        found = 0
                        for j in range(nB):
                            # Find d_ij
                            d_ij = np.subtract(candB[j][:2], candA[i][:2])
                            norm = np.linalg.norm(d_ij)
                            if norm:
                                d_ij = d_ij / norm
                            else:
                                continue
                            # Find p(u)
                            interp_coord = list(zip(np.linspace(candA[i][0], candB[j][0], num=n_interp_samples),
                                                    np.linspace(candA[i][1], candB[j][1], num=n_interp_samples)))
                            # Find L(p(u))
                            paf_interp = []
                            for k in range(len(interp_coord)):
                                paf_interp.append([pafA[int(round(interp_coord[k][1])), int(round(interp_coord[k][0]))],
                                                pafB[int(round(interp_coord[k][1])), int(round(interp_coord[k][0]))] ]) 
                            # Find E
                            paf_scores = np.dot(paf_interp, d_ij)
                            avg_paf_score = sum(paf_scores)/len(paf_scores)
                            
                            # Check if the connection is valid
                            # If the fraction of interpolated vectors aligned with PAF is higher then threshold -> Valid Pair  
                            if ( len(np.where(paf_scores > paf_score_th)[0]) / n_interp_samples ) > conf_th :
                                if avg_paf_score > maxScore:
                                    max_j = j
                                    maxScore = avg_paf_score
                                    found = 1
                        # Append the connection to the list
                        if found:            
                            valid_pair = np.append(valid_pair, [[candA[i][3], candB[max_j][3], maxScore]], axis=0)

                    # Append the detected connections to the global list
                    valid_pairs.append(valid_pair)
                else: # If no keypoints are detected
                    print("No Connection : k = {}".format(k))
                    invalid_pairs.append(k)
                    valid_pairs.append([])
            print(valid_pairs)
            return valid_pairs, invalid_pairs

        # This function creates a list of keypoints belonging to each person
        # For each detected valid pair, it assigns the joint(s) to a person
        # It finds the person and index at which the joint should be added. This can be done since we have an id for each joint
        def getPersonwiseKeypoints(valid_pairs, invalid_pairs):
            # the last number in each row is the overall score 
            personwiseKeypoints = -1 * np.ones((0, 19))

            for k in range(len(mapIdx)):
                if k not in invalid_pairs:
                    partAs = valid_pairs[k][:,0]
                    partBs = valid_pairs[k][:,1]
                    indexA, indexB = np.array(POSE_PAIRS[k])

                    for i in range(len(valid_pairs[k])): 
                        found = 0
                        person_idx = -1
                        for j in range(len(personwiseKeypoints)):
                            if personwiseKeypoints[j][indexA] == partAs[i]:
                                person_idx = j
                                found = 1
                                break

                        if found:
                            personwiseKeypoints[person_idx][indexB] = partBs[i]
                            personwiseKeypoints[person_idx][-1] += keypoints_list[partBs[i].astype(int), 2] + valid_pairs[k][i][2]

                        # if find no partA in the subset, create a new subset
                        elif not found and k < 17:
                            row = -1 * np.ones(19)
                            row[indexA] = partAs[i]
                            row[indexB] = partBs[i]
                            # add the keypoint_scores for the two keypoints and the paf_score 
                            row[-1] = sum(keypoints_list[valid_pairs[k][i,:2].astype(int), 2]) + valid_pairs[k][i][2]
                            personwiseKeypoints = np.vstack([personwiseKeypoints, row])
            return personwiseKeypoints

        # upload image code and processing on it
        uploaded_file = st.file_uploader("",type=["png","jpg","jpeg"])
        if uploaded_file is not None:
            # Convert the file to an opencv image.
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image1 = cv2.imdecode(file_bytes, 1)
            frameWidth = image1.shape[1]
            frameHeight = image1.shape[0]

            t = time.time()
            net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

            # Fix the input Height and get the width according to the Aspect Ratio
            inHeight = 368
            inWidth = int((inHeight/frameHeight)*frameWidth)

            inpBlob = cv2.dnn.blobFromImage(image1, 1.0 / 255, (inWidth, inHeight),
                                    (0, 0, 0), swapRB=False, crop=False)

            net.setInput(inpBlob)
            output = net.forward()
            # print("Time Taken = {}".format(time.time() - t))

            i = 0
            probMap = output[0, i, :, :]
            probMap = cv2.resize(probMap, (frameWidth, frameHeight))
            plt.figure(figsize=[14,10])
            plt.imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
            plt.imshow(probMap, alpha=0.6)
            plt.colorbar()
            plt.axis("off")

            
            detected_keypoints = []
            keypoints_list = np.zeros((0,3))
            keypoint_id = 0
            threshold = 0.1

            for part in range(nPoints):
                probMap = output[0,part,:,:]
                probMap = cv2.resize(probMap, (image1.shape[1], image1.shape[0]))
            #     plt.figure()
            #     plt.imshow(255*np.uint8(probMap>threshold))
                keypoints = getKeypoints(probMap, threshold)
                # print("Keypoints - {} : {}".format(keypointsMapping[part], keypoints))
                keypoints_with_id = []
                for i in range(len(keypoints)):
                    keypoints_with_id.append(keypoints[i] + (keypoint_id,))
                    keypoints_list = np.vstack([keypoints_list, keypoints[i]])
                    keypoint_id += 1

                detected_keypoints.append(keypoints_with_id)

            frameClone = image1.copy()
            for i in range(nPoints):
                for j in range(len(detected_keypoints[i])):
                    cv2.circle(frameClone, detected_keypoints[i][j][0:2], 3, [0,0,255], -1, cv2.LINE_AA)
            plt.figure(figsize=[15,15])
            plt.imshow(frameClone[:,:,[2,1,0]])

            valid_pairs, invalid_pairs = getValidPairs(output)
            personwiseKeypoints = getPersonwiseKeypoints(valid_pairs, invalid_pairs)

            for i in range(17):
                for n in range(len(personwiseKeypoints)):
                    index = personwiseKeypoints[n][np.array(POSE_PAIRS[i])]
                    if -1 in index:
                        continue
                    B = np.int32(keypoints_list[index.astype(int), 0])
                    A = np.int32(keypoints_list[index.astype(int), 1])
                    cv2.line(frameClone, (B[0], A[0]), (B[1], A[1]), colors[i], 3, cv2.LINE_AA)
            my_bar = st.progress(0)
            for percent_complete in range(100):
                time.sleep(0.1)
                my_bar.progress(percent_complete + 1)
            st.image(frameClone, channels="BGR")



    with pre:
        st.sidebar.markdown("<h1 style='color:Red ;'>References</h1>", unsafe_allow_html=True)
        st.sidebar.markdown("[Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields](https://arxiv.org/abs/1611.08050)")
        st.sidebar.markdown("[Understanding OpenPose](https://medium.com/analytics-vidhya/understanding-openpose-with-code-reference-part-1-b515ba0bbc73)")
        st.sidebar.markdown("[Multi Person Pose Estimation in OpenCV using OpenPose](https://learnopencv.com/multi-person-pose-estimation-in-opencv-using-openpose/)")
        st.sidebar.markdown("[How does Pose Estimation Work](https://www.youtube.com/watch?v=utz4Ql0CkBE&t=463s)")


        st.sidebar.markdown("<h2 style='color:Red ;'>If you like the Content Press Button and Enjoy: </h2>", unsafe_allow_html=True)
        if st.sidebar.button("Press if You Like"):
            st.balloons() 


else:
    header = st.beta_container()
    explain = st.beta_container()
    model = st.beta_container()
    pre = st.beta_container()


    with header:
        st.markdown("<h1 style='text-align: center; color:Red ;'>REAL TIME HUMAN POSE ESTIMATION: </h1>", unsafe_allow_html=True)


    with explain:
        st.header("Human Pose Estimation :person_with_ball:")
        st.write("Human body pose estimation from images or video plays a central role in various applications such as health tracking, sign language recognition, and gestural control. This task is challenging due to a wide variety of poses, numerous degrees of freedom, and occlusions")         
        st.write("Pose Estimation consists of three different blocks")
        st.write("1 Body+Foot detection")
        st.write("2 Hand detection")
        st.write("3 Face detection.")
        st.markdown("## Types of Pose Estimation ##")
        st.write("Single-Person Pose Estimation: (MediaPipe BlazePose)")
        st.write("Multi-Person Pose Estimation: (Openpose)")
        blaze = ['images/blazepose.gif', 'images/blazepose2.gif', 'images/blazepose3.gif']
        st.image(blaze, use_column_width=True* len(blaze))


        

    with model:
        st.markdown("<h2 style='text-align: center; color:Red ;'>Methods used for Human Pose Estimation </h2>", unsafe_allow_html=True)
        st.markdown("### 1) OPENPOSE : ### ")
        st.write("OpenPose is one of the most popular approache for multi-person human pose estimation.As with many bottom-up approaches, OpenPose first detects parts (key points) belonging to every person in the image, followed by assigning parts to distinct individuals.")
        st.image("images/openpose.png",width=450)
        st.markdown("### 2) BLAZEPOSE : ### ")
        st.write("BlazePose, a lightweight convolutional neural network architecture for human pose estimation that is tailored for real-time inference on mobile devices.During inference, the network produces 33 body keypoints for a single person and runs at around 30 frames per second . This makes it particularly suited to real-time use cases like fitness tracking and sign language recognition.Our main contributions include a novel body pose tracking solution and a lightweight body pose estimation neural network that uses both heatmaps and regression to keypoint coordinates.")
        st.image("images/wall.gif")

        
    with pre:
        st.sidebar.markdown("<h1 style='color:Red ;'>Requirements</h1>", unsafe_allow_html=True)
        st.sidebar.markdown("## [Python](https://www.python.org/downloads/)")
        st.sidebar.code('install python3 and greater')
        st.sidebar.markdown("## [Numpy](https://opencv.org/)")
        st.sidebar.code('$pip install opencv-python')
        st.sidebar.markdown("## [Pandas](https://pandas.pydata.org/)")
        st.sidebar.code('$pip install pandas')
        st.sidebar.markdown("## [scikit-learn](https://scikit-learn.org/stable/)")
        st.sidebar.code('$pip install scikit-learn')
        st.sidebar.markdown("## [Matplotlib](https://matplotlib.org/)")
        st.sidebar.code('$pip install matplotlib')
        st.sidebar.markdown("## [OpenCV](https://opencv.org/)")
        st.sidebar.code('$pip install opencv-python')
        st.sidebar.markdown("## [Mediapipe](https://mediapipe.dev/)")
        st.sidebar.code('$pip install mediapipe')
        st.sidebar.markdown("## [streamlit](https://streamlit.io/)")
        st.sidebar.code('$pip install streamlit')



    