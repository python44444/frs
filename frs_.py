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


def loc(known_face_imgs, face_img_to_check):
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
    plt.show()


def find(face, data):
    im = im(face, data)
    known_face_imgs = im[0]
    face_img_to_check = im[1]

    loc = loc(known_face_imgs, face_img_to_check)
    known_face_locs = loc[0]
    face_loc_to_check = loc[1]

    encoding = encoding(
        known_face_imgs, known_face_locs, face_img_to_check, face_loc_to_check
    )
    known_face_encodings = encoding[0]
    face_encoding_to_check = encoding[1]

    matches = face_recognition.compare_faces(
        known_face_encodings, face_encoding_to_check
    )
    print(matches)

    dists = face_recognition.face_distance(known_face_encodings, face_encoding_to_check)
    print(dists)

    # 検証用画像
    for img, loc in zip(known_face_imgs, known_face_locs):
        draw(img, loc)
    # 認識したい顔画像
    draw(face_img_to_check, face_loc_to_check)


face = "check.jpg"
data = "images/*"
find(face, data)
