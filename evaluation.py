from os.path import isfile, join
from os import listdir, stat
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from skimage.exposure import rescale_intensity
import numpy as np
import cv2
from matplotlib import pyplot as plt

from image_similarity_measures.quality_metrics import rmse, ssim, sre, psnr, fsim, issm, sam, uiq
from style_transfer import StyleTransfer
import os
import logging
import shutil

logging.basicConfig(level=logging.INFO, filename="logs/report.txt")


class EvaluationModel():
    def __init__(self):
        print('init')

    def generate_eigenface(self, img_path, num_com=10, raw_imgs=False, custom_path=None):
        inverts = []
        img = self.process_img(img_path, raw_imgs)
        faces_pca = PCA(num_com)

        for chann in img:
            faces_pca.fit_transform(chann)
            img_pca = faces_pca.transform(chann)
            img_inverted = faces_pca.inverse_transform(img_pca)
            inverts.append(img_inverted)

        path_name = custom_path if custom_path else f'eigenfaces/{img_path.split("/")[-1].split(".")[0]}.png'
        compressed = cv2.merge(inverts)
        cv2.imwrite(path_name, compressed)
        return path_name

    def process_img(self, img, raw=False):
        img = img if raw else cv2.imread(img)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (400, 400), interpolation=cv2.INTER_AREA)
        img = cv2.split(img)
        return img

    def transfer_styles(self, images):
        ref_src = "poses/makeup_hanna.jpeg"
        detail_pows = [0.2,  0.4,  0.6, 0.8, .9]
        blending_ratio = [0, 0.5, 1]
        dirs = []
        PCA_DIM = 10
        base_img_eigen, ref_eigen = None, None

        for img in images:
            try:
                print('Generating styles for ', img)

                base_path = img.split("/")[-1].split(".")[0]
                eigen_path, results_path = self.check_create_dir(base_path)

                s = StyleTransfer(img, ref_src)

                isolated_base_face, isolated_test_face = s.get_isolated_faces()
                """base_img_path = f'eigenfaces/{base_path}/original.png'
                base_img_eigen = self.generate_eigenface(
                    isolated_base_face, PCA_DIM, True, base_img_path)
                """
                ref_eigen_path = f'eigenfaces/{base_path}/example.png'
                ref_eigen = self.generate_eigenface(
                    isolated_test_face, PCA_DIM, True, ref_eigen_path)

                print('Generating styles')

                """for pw in detail_pows:
                    for ratio in blending_ratio:
                        results_path_name = self.gen_paths(
                            results_path, str(pw)+str(ratio))
                        eigen_path_name = self.gen_paths(
                            eigen_path, str(pw)+str(ratio))
                        s.transform(results_path_name, 1, .9 - pw, pw - 0.1)
                        eigen_dir = self.generate_eigenface(
                            results_path_name, PCA_DIM, False, eigen_path_name)"""

                # dirs.append(eigen_dir)
            except:
                print(f"Failed to generate styles for {img}")

            print('Style generation finished succesfully')
            """
            print("Checking for similarities between original image and reference images")
            self.check_sim([base_img_eigen], ref_eigen)

            print("Checking for similarities between original image and test images")

            self.check_sim(dirs, base_img_eigen)

            print("Checking for similarities between reference image and test images")

            self.check_sim(dirs, ref_eigen)
            """
    @staticmethod
    def gen_paths(path, identifier):
        return f'{path}/{identifier.replace(".", "")}.png'

    def check_sim(self):
        imgs = [
            f for f in listdir("eigenfaces")][44:500]

        print(imgs)
        imgs = list(filter(lambda x: os.path.exists(f"eigenfaces/{x}"), imgs))

        print(f"Items to check similarity for ", len(imgs))

        blendings = ['1']
        detail_pows = ['02',  '04',  '06', '08', '09']
        base_path = "eigenfaces/"
        results = []
        tested_cands = []

        idx = 0
        for img in imgs:
            try:
                print(f"Checking {idx}/{len(imgs)}")
                logging.info(f"Logging results for {img}")

                print(f"{base_path}{img}/original.png")
                orig_img_path = f"{base_path}{img}/original.png"
                example_img_path = f"{base_path}{img}/example.png"

                if os.path.exists(orig_img_path) and (example_img_path):
                    orig_img = cv2.imread(orig_img_path)
                    example_img = cv2.imread(example_img_path)

                    orig_exm = fsim(orig_img, example_img)
                    results.append([orig_exm])

                    origs = []
                    exms = []

                    for pw in detail_pows:
                        img2 = cv2.imread(
                            f"{base_path}{img}/{pw}1.png")

                        test_orig = fsim(img2, orig_img)
                        test_exm = fsim(img2, example_img)

                        origs.append(test_orig)
                        exms.append(test_exm)

                    results[idx].append([
                        origs,
                        exms,
                    ])

                    orig_txt = ".".join(str(v) for v in origs)
                    example_txt = ".".join(str(v) for v in exms)

                    logging.info(
                        f"Orig_sim={orig_exm}, original={orig_txt}, ex={example_txt}")
                    tested_cands.append(img)
                    idx += 1

            except Exception as e:
                print(f'wrong {e}')

        np.save('logs/data.npy', results)
        np.save('logs/images_cand.npy', tested_cands)

    @staticmethod
    def check_create_dir(path):
        eigen_path = f"eigenfaces/{path}"
        results_path = f"results/{path}"

        if not os.path.exists(eigen_path):
            os.mkdir(eigen_path)

        if not os.path.exists(results_path):
            os.mkdir(results_path)

        return eigen_path, results_path

    @staticmethod
    def clean_dirs():
        base_path = "eigenfaces/"
        imgs = [f for f in listdir("eigenfaces")][:500]

        for img in imgs:
            if os.path.exists(f"{base_path}{img}") and not os.path.exists(f"{base_path}{img}/020.png"):
                shutil.rmtree(f"{base_path}{img}")

    @staticmethod
    def process_evaluation():
        data = np.load("logs/data.npy", allow_pickle=True)

        base_sim = []
        dist_with_orig_1 = []
        dist_with_orig_2 = []
        dist_with_orig_3 = []
        dist_with_orig_4 = []
        dist_with_orig_5 = []
        dist_with_ref_1 = []
        dist_with_ref_2 = []

        for idx in range(0, len(data)):
            base, [orig, ref] = data[idx]

            if base > 0.5 and min(orig) > base:
                base_sim.append(base)
                dist_with_orig_1.append(orig[0])
                # dist_with_orig_2.append(orig[1])
                dist_with_orig_3.append(orig[2])
                # dist_with_orig_4.append(orig[3])
                dist_with_orig_5.append(orig[4])
                dist_with_ref_1.append(ref[0])
                dist_with_ref_2.append(ref[4])

        dist_with_orig_1 = np.array(dist_with_orig_1)
        dist_with_orig_5 = np.array(dist_with_orig_5)

        dist_with_ref_1 = np.array(dist_with_ref_1)
        dist_with_ref_2 = np.array(dist_with_ref_2)

        srted_1 = np.flip(np.argsort(dist_with_orig_5 -
                                     dist_with_orig_1))
        base_sim = np.array(base_sim)
        plt.figure(figsize=(20, 10))
        plt.plot(base_sim[srted_1], "+", markersize="3",
                 label="Distance between the original and the reference face")
        plt.vlines(range(0, len(base_sim)), ymin=dist_with_orig_1[srted_1],
                   ymax=dist_with_orig_5[srted_1], color="#542344", label="Distances between the transfer and the original face")
        plt.vlines(range(0, len(base_sim)), ymin=dist_with_ref_1[srted_1],
                   ymax=dist_with_ref_2[srted_1], color="#0F7173", label="Distances between the transfer and the referenced face")

        plt.plot(dist_with_orig_1[srted_1], ".", markersize=3, label="γ=0")
        plt.plot(dist_with_orig_5[srted_1], ".", markersize=3, label="γ=1")
        plt.plot(dist_with_ref_1[srted_1], ".", markersize=3, label="γ=0")
        plt.plot(dist_with_ref_2[srted_1], ".", markersize=3, label="γ=1")
        plt.ylim([0.3, 1])
        plt.ylabel("Distances")
        plt.xlabel('Candidates')
        plt.title('Feature based similarity index for n=400 candidates and α=1')
        plt.legend()
        plt.savefig("logs/experiment.png")
        # plt.show()

    @staticmethod
    def process_evaluation2():
        data = np.load("logs/data.npy", allow_pickle=True)

        base_sim = []
        dist_with_orig_1 = []
        dist_with_orig_2 = []
        dist_with_orig_3 = []
        dist_with_orig_4 = []
        dist_with_orig_5 = []
        dist_with_ref_1 = []
        dist_with_ref_2 = []
        dist_with_ref_3 = []
        dist_with_ref_4 = []
        dist_with_ref_5 = []

        for idx in range(0, len(data)):
            base, [orig, ref] = data[idx]

            if base > 0.5 and min(orig) > base:
                base_sim.append(base)
                dist_with_orig_1.append(orig[0])
                dist_with_orig_2.append(orig[1])
                dist_with_orig_3.append(orig[2])
                dist_with_orig_4.append(orig[3])
                dist_with_orig_5.append(orig[4])
                dist_with_ref_1.append(ref[0])
                dist_with_ref_2.append(ref[1])
                dist_with_ref_3.append(ref[2])
                dist_with_ref_4.append(ref[3])
                dist_with_ref_4.append(ref[4])

        dist_with_orig_1 = np.array(dist_with_orig_1)
        dist_with_orig_5 = np.array(dist_with_orig_5)

        dist_with_ref_1 = np.array(dist_with_ref_1)
        dist_with_ref_2 = np.array(dist_with_ref_2)
        dist_with_ref_3 = np.array(dist_with_ref_3)
        dist_with_ref_4 = np.array(dist_with_ref_4)

        def get_avg(x):
            return sum(x)/len(x)

        print(dist_with_ref_3, dist_with_ref_4, dist_with_ref_5)
        print((np.std(dist_with_ref_1) + np.std(dist_with_ref_2) + np.std(
            dist_with_ref_3) + np.std(dist_with_ref_4)) / 4)

    def generate_eigenfaces(self, imgs):
        for img in imgs:
            path = img.split('/')[-1]
            self.generate_eigenface(img, 10, False, f"final/eigenface_{path}")


e = EvaluationModel()
imgs = [
    f"data/test/{f}" for f in listdir("data/test") if isfile(join("data/test", f))]
e.process_evaluation2()
"""fig = plt.figure(figsize=(10, 3))
columns = 7
rows = 2
imgs = ["final/eigenface_warped.png", "final/eigenface_result0.png", "final/eigenface_result0.3.png",
        "final/eigenface_result0.6.png", "final/eigenface_result0.8.png", "final/eigenface_result1.png", "final/eigenface_isolated.png", "final/warped.png", "final/result0.png", "final/result0.3.png", "final/result0.6.png", "final/result0.8.png", "final/result1.png", "final/isolated.png"]

for idx, img in enumerate(reversed(imgs)):
    img = cv2.imread(img)
    fig.add_subplot(rows, columns, idx+1)
    plt.imshow(img[..., ::-1])
    plt.axis("off")

plt.savefig("final/eigenfaces_isolated.png")"""
