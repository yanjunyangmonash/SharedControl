import cv2
import os
import shutil


def get_frame_from_video(video_name, interval):

    # 保存图片的路径
    save_path = video_name.split('.mp4')[0] + '/'
    is_exists = os.path.exists(save_path)
    if not is_exists:
        os.makedirs(save_path)
        print('path of %s is build' % save_path)
    else:
        shutil.rmtree(save_path)
        os.makedirs(save_path)
        print('path of %s already exist and rebuild' % save_path)

    # Read video
    video_capture = cv2.VideoCapture(video_name)
    i = 0
    j = 0

    while True:
        success, frame = video_capture.read()
        i += 1
        if not success:
            print('video is all read')
            break
        elif i % interval == 0:
            # 保存图片
            j += 1
            #save_name = save_path + 'Dissection' + save_path[-2] + '_' + str(i) + '.jpg'
            save_name = 'E:/Clip30/clip30' + '_' + str(i) + '.jpg'
            cv2.imwrite(save_name, frame)
            print('image of %s is saved' % save_name)



if __name__ == '__main__':
    # 视频文件名字
    #for k in range(71, 81):
    #video_name = 'E:/cholec80/Surgeryvideo4150/ClippingCutting/Clip' + str(49) + '.mp4'
    video_name = 'E:/cholec80/Surgeryvideo2130/ClippingCutting/clip' + str(30) + '.mp4'
    interval = 1
    get_frame_from_video(video_name, interval)
    #print('Finish video' + str(k))
