from os import listdir
from os.path import join
import cv2
import numpy as np

# skin_db = [0] * 17000000
# n_skin_db = [0] * 17000000
red_skin = [0] * 256
green_skin = [0] * 256
blue_skin = [0] * 256
red_nskin = [0] * 256
green_nskin = [0] * 256
blue_nskin = [0] * 256
total_dict = {}


def get_files_from_path(i_path: str):
    img_files = listdir(i_path)
    img_file_paths = [join(i_path, f) for f in img_files]
    return img_file_paths


def get_img_data(skin_img_list: list, non_skin_img_list: list):
    total_images = len(skin_img_list)
    print(total_images)
    skin_total = 0
    nskin_total = 0
    for image_num in range(total_images):
        print("Image num = " + str(image_num))
        # img_skin = Image.open(skin_img_list[image_num])
        img_skin = cv2.imread(skin_img_list[image_num])
        img_non_skin = cv2.imread(non_skin_img_list[image_num])
        im_h, im_w, _ = img_skin.shape
        print(im_h, im_w)
        for h in range(im_h):
            for w in range(im_w):
                sb, sg, sr = img_skin[h, w]
                if sb > 225 and sg > 225 and sr > 225:
                    nskin_total += 1
                    nb, ng, nr = img_non_skin[h, w]
                    blue_nskin[nb] += 1
                    green_nskin[ng] += 1
                    red_nskin[nr] += 1
                    '''n_skin_idx = nb * 255 * 255 + ng * 255 + nr
                    n_skin_db[n_skin_idx] += 1'''
                else:
                    skin_total += 1
                    blue_skin[sb] += 1
                    green_skin[sg] += 1
                    red_skin[sr] += 1
                    '''skin_idx = sb * 255 * 255 + sg * 255 + sr
                    skin_db[skin_idx] += 1'''
    total_dict['skin'] = skin_total
    total_dict['nskin'] = nskin_total


def convert_img(img_test, skin_np_arr, nskin_np_arr):
    img_tt = cv2.imread(img_test)
    im_h, im_w, _ = img_tt.shape
    for h in range(im_h):
        for w in range(im_w):
            b, g, r = img_tt[h, w]
            idx = b * 255 * 255 + g * 255 + r
            if nskin_np_arr[idx] == 0:
                if skin_np_arr[idx] == 0:
                    continue
                else:
                    img_tt[h, w] = (0, 0, 0)
            else:
                ratio = skin_np_arr[idx] / nskin_np_arr[idx]
                if ratio > .4:
                    img_tt[h, w] = (0, 0, 0)
    cv2.imwrite("Result_" + img_test, img_tt)
    pass

def convert_img_cond(bb, gg, rr):
    np_b_ns = np.load("np_b_ns.npy")
    np_b_s = np.load("np_b_s.npy")
    np_g_ns = np.load("np_g_ns.npy")
    np_g_s = np.load("np_g_s.npy")
    np_r_ns = np.load("np_r_ns.npy")
    np_r_s = np.load("np_r_s.npy")
    np_total = np.load("np_total.npy")

    skin_t = int(np_total[0][1])
    nskin_t = int(np_total[1][1])
    skin_prob = skin_t / (skin_t + nskin_t)
    nskin_prob = nskin_t / (skin_t + nskin_t)
    #print(bb, gg, rr)
    #print(np_b_s[bb], np_g_s[gg], np_r_s[rr])
    np_b_ps = np_b_s / skin_t
    np_b_pns = np_b_ns / nskin_t
    np_g_ps = np_g_s / skin_t
    np_g_pns = np_g_ns / nskin_t
    np_r_ps = np_r_s / skin_t
    np_r_pns = np_r_ns / nskin_t

    prob_skin = skin_prob * float(np_b_ps[bb]) * float(np_g_ps[gg]) * float(np_r_ps[rr])
    prob_nskin = nskin_prob * float(np_b_pns[bb]) * float(np_g_pns[gg]) * float(np_r_pns[rr])
    #print(prob_skin)
    #print(prob_nskin)
    norm_skin = prob_skin/(prob_skin+prob_nskin)
    norm_nskin = prob_nskin / (prob_nskin + prob_nskin)

    #print(norm_skin, norm_nskin)
    if norm_skin>norm_nskin:
        return True
    else:
        return False


if __name__ == '__main__':
    '''skin_img_path = "D:\ibtd\ibtd\Mask"
    skin_images = get_files_from_path(skin_img_path)
    non_skin_path = "D:\ibtd\ibtd\Test"
    non_skin_images = get_files_from_path(non_skin_path)
    get_img_data(skin_images, non_skin_images)

    np_r_s = np.array(red_skin)
    np_r_ns = np.array(red_nskin)
    np_g_s = np.array(green_skin)
    np_g_ns = np.array(green_nskin)
    np_b_s = np.array(blue_skin)
    np_b_ns = np.array(blue_nskin)
    result = total_dict.items()

    # Convert object to a list
    data = list(result)
    np_total = np.array(data)


    np.save('np_r_s', np_r_s)
    np.save('np_r_ns', np_r_ns)
    np.save('np_g_s', np_g_s)
    np.save('np_g_ns', np_g_ns)
    np.save('np_b_s', np_b_s)
    np.save('np_b_ns', np_b_ns)
    np.save('np_total', np_total)'''


    sample_img = "multiple_person.PNG"
    sample_img_c = cv2.imread(sample_img)
    im_h, im_w, _ = sample_img_c.shape
    for h in range(im_h):
        for w in range(im_w):
            #print(h, w)
            b, g, r = sample_img_c[h, w]
            s = convert_img_cond(b, g, r)
            if s:
                sample_img_c[h, w] = (0, 0, 0)
    cv2.imwrite("Result_" + sample_img, sample_img_c)
    #convert_img_cond(rr, gg, bb)

    #total_skin_prob = skin_t/nskin_t
    #total_nskin_prob = nskin_t/ (skin_t+nskin_t)
    '''print(total_skin_prob, total_nskin_prob)'''
    # convert_img(sample_img, skin_np_prob, nskin_np_prob)
