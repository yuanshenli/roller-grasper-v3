import numpy as np
import math
import random
import torch
import time
import os
from torch import nn
import torch.utils.tensorboard as tb
from utils import *
from config import *

from net import Net


class CollisionOracle(object):
    def __init__(self, raw_point_cloud_file, batch_size=576, lr=1e-5,
                 save_top_dir="collision_oracle", model_dir="model", log_writer="runs_log", device=torch.device("cpu")):
        self.device = device
        self.raw_point_cloud = np.load(raw_point_cloud_file)
        self.n_points = self.raw_point_cloud.shape[0]
        self.batch_size = batch_size
        self.lr = lr
        self.net = Net(device=device).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)

        self.save_top_dir = save_top_dir
        if not os.path.exists(self.save_top_dir):
            os.makedirs(self.save_top_dir)
        self.model_dir = os.path.join(self.save_top_dir, model_dir)
        self.writer_dir = os.path.join(self.save_top_dir, log_writer)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        if not os.path.exists(self.writer_dir):
            os.makedirs(self.writer_dir)

        self.pos_weight = 1.

        self.writer = tb.SummaryWriter(self.writer_dir)

    def predict(self, x):
        if x.ndim == 1:
            x = x.reshape(1, -1)
        x = torch.FloatTensor(
            self.point_cloud_represent(self.raw_point_cloud, x)).to(self.device)
        with torch.no_grad():
            self.net.eval()
            score = self.net(x)
            self.net.train()
        return score.cpu().data.numpy().flatten()

    def train(self, n_steps=int(1e6), threshs=(0.5, 0.75)):
        min_validation_loss, max_validation_accuracy = float("inf"), 0
        # thresh_1, thresh_2 = 0.5, 0.75
        eval_no = 0
        for i in range(n_steps):
            loss_func = nn.BCELoss()

            indices_pos = self.training_pos_indices[np.random.choice(self.training_pos_cnt, size=self.batch_size // 2)]
            indices_neg = self.training_neg_indices[np.random.choice(self.training_neg_cnt, size=self.batch_size // 2)]

            batch_X = np.vstack((self.training_X[indices_pos, :], self.training_X[indices_neg, :]))
            batch_y = np.vstack((self.training_y[indices_pos, :], self.training_y[indices_neg, :]))

            batch_X_new = torch.FloatTensor(
                self.point_cloud_represent(self.raw_point_cloud, batch_X)).to(self.device)
            batch_y = torch.FloatTensor(batch_y).to(self.device)

            y_predict = self.net(batch_X_new)
            loss = loss_func(y_predict, batch_y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            train_loss = loss.cpu().data.item()
            self.writer.add_scalar('training/loss', train_loss, i)

            pos_cnt = self.batch_size // 2
            neg_cnt = self.batch_size // 2

            for th in threshs:
                correct_cnt = ((y_predict > th).float() == batch_y).float().sum().item()
                train_acc = correct_cnt / batch_y.size(0)
                self.writer.add_scalar('training/accuracy_thresh_' + str(th) + '_', train_acc, i)

                tp_cnt = ((y_predict > th) * (batch_y == 1)).sum().item()
                tn_cnt = ((y_predict <= th) * (batch_y == 0)).sum().item()
                tp_rate = tp_cnt / pos_cnt
                fp_rate = (neg_cnt - tn_cnt) / neg_cnt
                tn_rate = tn_cnt / neg_cnt
                fn_rate = (pos_cnt - tp_cnt) / pos_cnt
                precision = 0 if tp_cnt == 0 else tp_cnt / (tp_cnt + neg_cnt - tn_cnt)
                self.writer.add_scalar('training/precision_thresh_' + str(th) + '_', precision, i)
                self.writer.add_scalar('training/TPR_thresh_' + str(th) + '_', tp_rate, i)
                self.writer.add_scalar('training/FPR_thresh_' + str(th) + '_', fp_rate, i)
                self.writer.add_scalar('training/TNR_thresh_' + str(th) + '_', tn_rate, i)
                self.writer.add_scalar('training/FNR_thresh_' + str(th) + '_', fn_rate, i)

            if (i + 1) % 20000 == 0:

                val_loss, val_acc, val_precision, tp_rate, fp_rate, tn_rate, fn_rate = self.evaluate(threshs=threshs)

                for th in threshs:
                    self.writer.add_scalar('validation/loss', val_loss[th], i)
                    self.writer.add_scalar('validation/accuracy_thresh_' + str(th) + '_', val_acc[th], i)
                    self.writer.add_scalar('validation/precision_thresh_' + str(th) + '_', val_precision[th], i)
                    self.writer.add_scalar('validation/TPR_thresh_' + str(th) + '_', tp_rate[th], i)
                    self.writer.add_scalar('validation/FPR_thresh_' + str(th) + '_', fp_rate[th], i)
                    self.writer.add_scalar('validation/TNR_thresh_' + str(th) + '_', tn_rate[th], i)
                    self.writer.add_scalar('validation/FNR_thresh_' + str(th) + '_', fn_rate[th], i)

                    log("Step {} thresh {}: TP {:.4f}, FP {:.4f}, TN {:.4f}, FN {:.4f}".format(
                        i + 1, th, tp_rate[th], fp_rate[th], tn_rate[th], fn_rate[th]), 'b', 'B')
                save_flag = False
                if save_flag or (i + 1) % 10000 == 0:
                    self.save_model(i + 1)
                eval_no += 1

    def evaluate(self, threshs=(0.5, 0.75), batch_size=2048):

        n_samples, pos_cnt, neg_cnt = 0, 0, 0
        correct_cnt, tp_cnt, fp_cnt, tn_cnt, fn_cnt = {}, {}, {}, {}, {}
        val_loss_seq = {}
        for th in threshs:
            correct_cnt[th], tp_cnt[th], fp_cnt[th], tn_cnt[th], fn_cnt[th] = 0, 0, 0, 0, 0
            val_loss_seq[th] = []
        for i in range(self.validation_y.shape[0] // batch_size):
            validation_X = self.validation_X[i * batch_size: (i + 1) * batch_size, :]
            validation_y = self.validation_y[i * batch_size: (i + 1) * batch_size, :]

            validation_X_new = torch.FloatTensor(
                self.point_cloud_represent(self.raw_point_cloud, validation_X)).to(self.device)
            validation_y = torch.FloatTensor(validation_y).to(self.device)

            loss_func = nn.BCELoss()
            with torch.no_grad():
                self.net.eval()
                y_predict = self.net(validation_X_new)
                self.net.train()
                for th in threshs:
                    correct_cnt[th] += ((y_predict > th).float() == validation_y).float().sum().item()
                    tp_cnt[th] += ((y_predict > th) * (validation_y == 1)).sum().item()
                    tn_cnt[th] += ((y_predict <= th) * (validation_y == 0)).sum().item()
                    val_loss = loss_func(y_predict, validation_y).item()
                    val_loss_seq[th].append(val_loss)
                pos_cnt += (validation_y == 1).float().sum().item()
                neg_cnt += (validation_y == 0).float().sum().item()
                n_samples += validation_y.size(0)
                assert n_samples == pos_cnt + neg_cnt

        mean_val_loss, val_accuracy, precision, tp_rate, fp_rate, tn_rate, fn_rate = {}, {}, {}, {}, {}, {}, {}
        for th in threshs:
            val_accuracy[th] = correct_cnt[th] / n_samples
            tp_rate[th] = 0 if pos_cnt == 0 else tp_cnt[th] / pos_cnt
            fp_rate[th] = 0 if neg_cnt == 0 else (neg_cnt - tn_cnt[th]) / neg_cnt
            tn_rate[th] = 0 if neg_cnt == 0 else tn_cnt[th] / neg_cnt
            fn_rate[th] = 0 if pos_cnt == 0 else (pos_cnt - tp_cnt[th]) / pos_cnt
            precision[th] = 0 if tp_cnt[th] == 0 else tp_cnt[th] / (tp_cnt[th] + neg_cnt - tn_cnt[th])
            mean_val_loss[th] = np.mean(val_loss_seq[th])
        return mean_val_loss, val_accuracy, precision, tp_rate, fp_rate, tn_rate, fn_rate

    def save_model(self, step):
        save_path = os.path.join(self.model_dir, str(step))
        torch.save(self.net.state_dict(), save_path + "_actor")
        torch.save(self.optimizer.state_dict(), save_path + "_actor_optimizer")
        print("Saving model at step %d to %s" % (step, save_path))

    def restore(self, step):
        load_path = os.path.join(self.model_dir, str(step))
        self.net.load_state_dict(torch.load(load_path + "_actor", map_location=self.device))
        self.optimizer.load_state_dict(torch.load(load_path + "_actor_optimizer", map_location=self.device))

    def point_cloud_represent(self, raw_point_cloud, frame_pair):
        """
        raw_point_cloud: (n_points, 3)
        frame_pair: (B, 14)

        return: (B, 12, n_points)
                12d: (recentered point cloud at start pose,
                      recentered point cloud at goal pose,
                      point cloud center at start pose,
                      point cloud center at goal pose)
        """
        batch_size = frame_pair.shape[0]
        frame_pair_split = np.hsplit(frame_pair, [7])
        start_frame, end_frame = frame_pair_split[0].copy(), frame_pair_split[1].copy()
        start_pc = self.trans_point_cloud(raw_point_cloud, start_frame)  # (B, 3, n_points)
        end_pc = self.trans_point_cloud(raw_point_cloud, end_frame)  # (B, 3, n_points)
        start_center = np.expand_dims(start_frame[:, :3], axis=2)  # (B, 3, 1)
        start_center[:, 2, :] -= 0.2
        start_center *= 10.0
        end_center = np.expand_dims(end_frame[:, :3], axis=2)  # (B, 3, 1)
        end_center[:, 2, :] -= 0.2
        end_center *= 10.0
        center = np.concatenate((start_center, end_center), axis=1)  # (B, 6, 1)
        pc_sg = np.concatenate((start_pc, end_pc), axis=1)  # (B, 6, n_points)
        x_pc = pc_sg - center  # (B, 6, n_points)
        center = np.tile(center, (1, 1, self.n_points))  # (B, 6, n_points)

        return np.concatenate((x_pc, center), axis=1)  # (B, 12, n_points)

    def trans_point_cloud(self, point_cloud, frame, rotation_represent="rot_vec"):
        if rotation_represent == "rot_vec":
            pos, quat = frame[:, :3], rotvec_to_quat_batch(frame[:, 3:6], frame[:, 6].reshape(frame.shape[0], 1))
        elif rotation_represent == "quat":
            pos, quat = frame[:, :3], frame[:, -4:]
        else:
            raise ValueError
        rot_mat = R.from_quat(mjcquat_to_sciquat_batch(quat)).as_matrix()  # (B, 3, 3)

        T_mat = np.zeros((frame.shape[0], 4, 4))  # (B, 4, 4)
        T_mat[:, :3, :3] = rot_mat
        T_mat[:, :3, 3] = pos
        T_mat[:, 3, 3] = 1

        point_cloud = np.hstack((point_cloud, np.ones((point_cloud.shape[0], 1))))  # (n_points, 4)
        point_cloud_new = (T_mat @ point_cloud.T)[:, :3, :]  # (B, 3, n_points)
        point_cloud_new[:, 2, :] -= 0.2
        point_cloud_new *= 10.0
        return point_cloud_new  # shape (B, 3, n_points)

    def set_point_cloud(self, point_cloud):
        self.raw_point_cloud = point_cloud.copy()
        self.n_points = point_cloud.shape[0]

    def load_data(self, training_set_file="train.npy", validation_set_file="val.npy"):
        print("load training set from", training_set_file)
        print("load validation set from", validation_set_file)
        training_set = np.load(training_set_file)
        validation_set = np.load(validation_set_file)
        self.validation_X = validation_set[:, :-1]
        self.validation_y = validation_set[:, [-1]].reshape(-1, 1)
        self.training_X = training_set[:, :-1]
        self.training_y = training_set[:, [-1]].reshape(-1, 1)
        self.training_pos_indices = np.where(self.training_y[:, -1] == 1)[0]
        self.training_neg_indices = np.where(self.training_y[:, -1] == 0)[0]
        self.training_pos_cnt = self.training_pos_indices.shape[0]
        self.training_neg_cnt = self.training_neg_indices.shape[0]
        log("training set: {} positive, {} negative".format(self.training_pos_cnt, self.training_neg_cnt))

        print("val_X", self.validation_X.shape, "val_y", self.validation_y.shape,
              "train_X", self.training_X.shape, "train_y", self.training_y.shape)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="exp_collision", help="directory to save experiment output")
    parser.add_argument("--dataset_dir", type=str, default="data/collision_dataset",
                        help="directory of collision dataset")
    parser.add_argument("--obj_type", type=str, help="object type",
                        choices=["cube", "rectangular", "cube_w_opening", "mug"])
    parser.add_argument("--train_steps", type=int, default=int(1e6), help="number of training steps")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    args = parser.parse_args()

    np.random.seed(args.seed)

    raw_point_cloud_file = f"point_cloud/{args.obj_type}.npy"
    collision_oracle = CollisionOracle(raw_point_cloud_file=raw_point_cloud_file,
                                       device=device, save_top_dir=args.output_dir)
    collision_oracle.load_data(
        training_set_file=f"{args.dataset_dir}/{args.obj_type}/train.npy",
        validation_set_file=f"{args.dataset_dir}/{args.obj_type}/val.npy"
    )

    collision_oracle.train(args.train_steps)
