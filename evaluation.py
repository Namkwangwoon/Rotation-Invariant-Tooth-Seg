import numpy as np

def cal_metric(gt_labels, pred_sem_labels, pred_ins_labels, is_half=None, vertices=None):
    ins_label_names = np.unique(pred_ins_labels)
    ins_label_names = ins_label_names[ins_label_names != 0]
    IOU = 0
    F1 = 0
    ACC = 0
    SEM_ACC = 0
    IOU_arr = []
    for ins_label_name in ins_label_names:
        #instance iou
        ins_label_name = int(ins_label_name)
        ins_mask = pred_ins_labels==ins_label_name
        gt_label_uniqs, gt_label_counts = np.unique(gt_labels[ins_mask], return_counts=True)
        gt_label_name = gt_label_uniqs[np.argmax(gt_label_counts)]
        gt_mask = gt_labels == gt_label_name

        TP = np.count_nonzero(gt_mask * ins_mask)
        FN = np.count_nonzero(gt_mask * np.invert(ins_mask))
        FP = np.count_nonzero(np.invert(gt_mask) * ins_mask)
        TN = np.count_nonzero(np.invert(gt_mask) * np.invert(ins_mask))

        ACC += (TP + TN) / (FP + TP + FN + TN)
        precision = TP / (TP+FP)
        recall = TP / (TP+FN)
        F1 += 2*(precision*recall) / (precision + recall)
        IOU += TP / (FP+TP+FN)
        IOU_arr.append(TP / (FP+TP+FN))
        #segmentation accuracy
        pred_sem_label_uniqs, pred_sem_label_counts = np.unique(pred_sem_labels[ins_mask], return_counts=True)
        sem_label_name = pred_sem_label_uniqs[np.argmax(pred_sem_label_counts)]
        if is_half:
            if sem_label_name == gt_label_name or sem_label_name + 8 == gt_label_name:
                SEM_ACC +=1
        else:
            if sem_label_name == gt_label_name:
                SEM_ACC +=1
        #print("gt is", gt_label_name, "pred is", sem_label_name, sem_label_name == gt_label_name)
    if len(ins_label_names) == 0:
        print("zero labels!")
        ins_label_names = [0]
    return IOU/len(ins_label_names), F1/len(ins_label_names), ACC/len(ins_label_names), SEM_ACC/len(ins_label_names), IOU_arr