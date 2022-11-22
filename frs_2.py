import face_recognition
import matplotlib.pyplot as plt
import glob


def im(face, data):
    known_face_imgs = []
    paths = glob.glob(data)

    for path in paths:
        img = face_recognition.load_image_file(path)
        known_face_imgs.append(img)

    face_img_to_check = face_recognition.load_image_file(face)

    return known_face_imgs, face_img_to_check


def check(known_face_imgs, face_img_to_check):
    known_face_locs = []
    for img in known_face_imgs:
        loc = face_recognition.face_locations(img, model="hog")
        assert len(loc) == 1, "画像から顔の検出に失敗したか、2人以上の顔が検出されました"
        known_face_locs.append(loc)

    face_loc_to_check = face_recognition.face_locations(face_img_to_check, model="hog")
    assert len(face_loc_to_check) == 1, "画像から顔の検出に失敗したか、2人以上の顔が検出されました"

    return known_face_locs, face_loc_to_check


def encoding(known_face_imgs, known_face_locs, face_img_to_check, face_loc_to_check):
    known_face_encodings = []
    for img, loc in zip(known_face_imgs, known_face_locs):
        (encoding,) = face_recognition.face_encodings(img, loc)
        known_face_encodings.append(encoding)

    (face_encoding_to_check,) = face_recognition.face_encodings(
        face_img_to_check, face_loc_to_check
    )

    return known_face_encodings, face_encoding_to_check


def draw(img, locations):
    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.set_axis_off()
    for i, (top, right, bottom, left) in enumerate(locations):
        w, h = right - left, bottom - top
        ax.add_patch(plt.Rectangle((left, top), w, h, ec="r", lw=2, fill=None))
        ax.add_patch(plt.text(x=10, y=10, s="text", fontsize="xx-large"))
    plt.show()


def find(face, data):
    known_face_imgs, face_img_to_check = im(face, data)

    known_face_locs, face_loc_to_check = check(known_face_imgs, face_img_to_check)

    known_face_encodings, face_encoding_to_check = encoding(
        known_face_imgs, known_face_locs, face_img_to_check, face_loc_to_check
    )

    # 探したい人間の顔かどうかTrueもしくはFalseで表される
    matches = face_recognition.compare_faces(
        known_face_encodings, face_encoding_to_check
    )
    print(matches)
    # 探したい人間の顔にどれだけ近いかを示す、数字が0に近いほど一致率は高いといえる
    dists = face_recognition.face_distance(known_face_encodings, face_encoding_to_check)
    print(dists)

    # 認識したい人の顔画像
    draw(face_img_to_check, face_loc_to_check)
    # 検証用画像
    for img, loc in zip(known_face_imgs, known_face_locs):
        draw(img, loc)


face = "check.jpg"
data = "images/*"
find(face, data)
