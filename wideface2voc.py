import os,cv2,sys,shutil

from xml.dom.minidom import Document

def writexml(filename,saveimg,bboxes,xmlpath,typename):
    #xml打包的函数，我们不需要知道内部做了什么。
    #我们只需要将图片名称   图片信息   bbox信息    最终存储信息 作为参数 就可以了
    #不需要做修改

    doc = Document()                                #定义文件对象
    annotation = doc.createElement('annotation')  #创建根节点
    doc.appendChild(annotation)    #存放在doc中
    # 定义annotation 的子节点
    folder = doc.createElement('folder')

    folder_name = doc.createTextNode('widerface')
    folder.appendChild(folder_name)
    annotation.appendChild(folder)
    filenamenode = doc.createElement('filename')
    filename_name = doc.createTextNode(filename)
    filenamenode.appendChild(filename_name)
    annotation.appendChild(filenamenode)
    source = doc.createElement('source')
    annotation.appendChild(source)
    database = doc.createElement('database')
    database.appendChild(doc.createTextNode('wider face Database'))
    source.appendChild(database)
    annotation_s = doc.createElement('annotation')
    annotation_s.appendChild(doc.createTextNode('PASCAL VOC2007'))
    source.appendChild(annotation_s)
    image = doc.createElement('image')
    image.appendChild(doc.createTextNode('flickr'))
    source.appendChild(image)
    flickrid = doc.createElement('flickrid')
    flickrid.appendChild(doc.createTextNode('-1'))
    source.appendChild(flickrid)
    owner = doc.createElement('owner')
    annotation.appendChild(owner)
    flickrid_o = doc.createElement('flickrid')
    flickrid_o.appendChild(doc.createTextNode('TAO'))
    owner.appendChild(flickrid_o)
    name_o = doc.createElement('name')
    name_o.appendChild(doc.createTextNode('TAO'))
    owner.appendChild(name_o)

    size = doc.createElement('size')
    annotation.appendChild(size)

    width = doc.createElement('width')
    width.appendChild(doc.createTextNode(str(saveimg.shape[1])))
    height = doc.createElement('height')
    height.appendChild(doc.createTextNode(str(saveimg.shape[0])))
    depth = doc.createElement('depth')
    depth.appendChild(doc.createTextNode(str(saveimg.shape[2])))

    size.appendChild(width)

    size.appendChild(height)
    size.appendChild(depth)
    segmented = doc.createElement('segmented')
    segmented.appendChild(doc.createTextNode('0'))
    annotation.appendChild(segmented)
    for i in range(len(bboxes)):
        bbox = bboxes[i]
        objects = doc.createElement('object')
        annotation.appendChild(objects)
        object_name = doc.createElement('name')
        object_name.appendChild(doc.createTextNode('face'))  #人脸数据的话 直接为 “face”
        objects.appendChild(object_name)
        pose = doc.createElement('pose')
        pose.appendChild(doc.createTextNode('Unspecified'))
        objects.appendChild(pose)
        truncated = doc.createElement('truncated')
        truncated.appendChild(doc.createTextNode('1'))
        objects.appendChild(truncated)
        difficult = doc.createElement('difficult')
        difficult.appendChild(doc.createTextNode('0'))
        objects.appendChild(difficult)
        bndbox = doc.createElement('bndbox')
        objects.appendChild(bndbox)
        xmin = doc.createElement('xmin')
        xmin.appendChild(doc.createTextNode(str(bbox[0])))
        bndbox.appendChild(xmin)
        ymin = doc.createElement('ymin')
        ymin.appendChild(doc.createTextNode(str(bbox[1])))
        bndbox.appendChild(ymin)
        xmax = doc.createElement('xmax')
        xmax.appendChild(doc.createTextNode(str(bbox[2])))#  bbox[0] +
        bndbox.appendChild(xmax)
        ymax = doc.createElement('ymax')
        ymax.appendChild(doc.createTextNode(str(bbox[3])))#  bbox[1] +
        bndbox.appendChild(ymax)
    f = open(xmlpath, "w")
    f.write(doc.toprettyxml(indent=''))
    f.close()


rootdir = "E:/python-study/DATA_PREPARE/wider_face"  #定义数据集的根目录wider_face  下载好的


def convertimgset(img_set):  #解析函数 img_set 作为解析的路径    img_sets = ["train","val"]
    imgdir = rootdir + "/WIDER_" + img_set + "/images"                                   #图片文件的路径
    gtfilepath = rootdir + "/wider_face_split/wider_face_" + img_set + "_bbx_gt.txt"  #标注信息
    fwrite = open(rootdir + "/ImageSets/Main/" + img_set + ".txt", 'w')  #写入txt中 main 底下的文件夹 对应140行
    index = 0    #表示解析到第几张图
    with open(gtfilepath, 'r') as gtfiles:      #打开真值文件，获取bbox
        while(True):                            #true   index< 1000 #前1000个样本
            filename = gtfiles.readline()[:-1]
            print(filename)#读取一行数据， 为图像路径
            if filename == None or filename == "":
                break
            imgpath = imgdir + "/" + filename         #图片的绝对路径
            img = cv2.imread(imgpath)                 #拿到读取图片   可以获取到shape信息
            if not img.data:
                break
            numbbox = int(gtfiles.readline())         #读取到了第二行    人脸个数
            bboxes = []
            print(numbbox)
            for i in range(numbbox):                 #读取bbox信息  numbbox 行
                line = gtfiles.readline()
                lines = line.split(" ")
                lines = lines[0:4]

                bbox = (int(lines[0]), int(lines[1]), int(lines[0])+int(lines[2]), int(lines[1])+int(lines[3]))  #存储的左上角 坐标 和 高度宽度

                if int(lines[2]) < 40 or int(lines[3]) < 40:
                    continue

                bboxes.append(bbox)             #存放到bbox中   numbbox个人脸信息

                #cv2.rectangle(img, (bbox[0],bbox[1]),(bbox[0]+bbox[2],bbox[1]+bbox[3]),color=(255,255,0),thickness=1)

            filename = filename.replace("/", "_")    #图片的名称存储

            if len(bboxes) == 0:
                print("no face")
                continue
            #cv2.imshow("img", img)
            #cv2.waitKey(0)
            cv2.imwrite("{}/JPEGImages/{}".format(rootdir,filename), img)  #写入图像JPEGImages
            fwrite.write(filename.split(".")[0] + "\n")     #写入txt中 main 底下的文件夹
            xmlpath = "{}/Annotations/{}.xml".format(rootdir,filename.split(".")[0])
            writexml(filename, img, bboxes, xmlpath, 'face')   #调用函数
            print("success number is ", index)
            index += 1

    fwrite.close()

if __name__=="__main__":
    img_sets = ["train","val"]
    for img_set in img_sets:
        convertimgset(img_set)
    #修改文件名
    shutil.move(rootdir + "/ImageSets/Main/" + "train.txt", rootdir + "/ImageSets/Main/" + "trainval.txt")
    shutil.move(rootdir + "/ImageSets/Main/" + "val.txt", rootdir + "/ImageSets/Main/" + "test.txt")