import cv2, math, time

# computes diagonal distance
def pythagorean(x, y):
    return math.sqrt(x ** 2 + y ** 2)

# finds minimum x and maximum y
def finger_coordinates(contour_list, frame):
    x = contour_list[0][0]
    y = contour_list[0][1]
    h = contour_list[0][3]
    for contour in contour_list:
        p_new = pythagorean(frame.shape[1] - contour[0], frame.shape[0] - contour[1])
        if p_new > pythagorean(frame.shape[1] - x, frame.shape[0] - y):
            x = contour[0]
            y = contour[1]
            h = contour[3]
    return (x, y, h)

# draws multiple lines by connecting dots
def draw_multidotted_line(coords, frame):
    length = len(coords)
    for i in range(length):
        if i != length - 1:
            cv2.line(frame, coords[i], coords[i + 1], (0, 255, 0), 5)

if __name__ == '__main__':
    # filiming requirement: self-cam, vertical, motion starts from bottom left (will add more angles later)
    video_path = '/Users/JingyuCai/Documents/Python/hand1.MOV'
    video = cv2.VideoCapture(video_path)
    first_frame = None
    coords = []
    while True:
        contour_list = []
        check, frame = video.read()
        print(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        if first_frame is None:
            first_frame = gray
            continue
        delta_frame = cv2.absdiff(first_frame, gray)
        thresh_delta = cv2.threshold(delta_frame, 50, 255, cv2.THRESH_BINARY)[1]
        thresh_delta = cv2.dilate(thresh_delta, None, iterations = 0)
        (_, cnts, _) = cv2.findContours(thresh_delta.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in cnts:
            if cv2.contourArea(contour) < 1000:
                continue
            (x, y, w, h) = cv2.boundingRect(contour)
            contour_list.append([x, y, w, h])
        if len(contour_list) != 0:
            (x, y, h) = finger_coordinates(contour_list, frame)
            # cv2.circle(frame, (x, y + h), 5, (0, 255, 0), 10)
            coords.append((x, y + h))
            draw_multidotted_line(coords, frame)
        frame = cv2.resize(frame, (int(frame.shape[1] / 2), int(frame.shape[0] / 2)))
        cv2.imshow('gesturer', frame)
        # cv2.imshow('capturing', gray)
        # cv2.imshow('delta', delta_frame)
        # cv2.imshow('thresh', thresh_delta)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    video.release()
    cv2.destroyAllWindows()