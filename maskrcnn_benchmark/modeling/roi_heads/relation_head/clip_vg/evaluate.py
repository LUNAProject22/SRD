import numpy as np
# Evluate recall at 20, 50, 100 for predicate relations
# calculate recall for each relation category
# accumulate across all images
class evaluate_recall(object):
    def __init__(self, num_classes=51, topk=20):
        self.topk = topk
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.correct = [0] * self.num_classes
        self.total = [0] * self.num_classes

    def update(self, gt_relations, Detected_List):
        # gt_relations: np.array([[ 1,  5, 20], [ 1, 11, 20], [ 1, 10, 20], [ 1, 13, 20], [ 1, 14, 20], [15,  6, 29]])
        # Detected_List: [[subject_index_1, object_index_1, relation_index_1, confidence_1], [subject_index_2, object_index_2, relation_index_2, confidence_2], ...]
        # select top k from Detected_List
        Detected_List = Detected_List[:self.topk]
        # Convert to list of strings
        detected_relations_str = []
        for i in range(len(Detected_List)):
            detected_relation = Detected_List[i]
            detected_subject = detected_relation[0]
            detected_object = detected_relation[1]
            detected_predicate = detected_relation[2]
            detected_string = "{}-{}-{}".format(detected_subject, detected_object, detected_predicate)
            detected_relations_str.append(detected_string)

        for i in range(len(gt_relations)):
            gt_relation = gt_relations[i]
            gt_subject = gt_relation[0]
            gt_object = gt_relation[1]
            gt_predicate = gt_relation[2]
            gt_string = "{}-{}-{}".format(gt_subject, gt_object, gt_predicate)

            # check gt_string in detected_relations_str
            if gt_string in detected_relations_str:
                self.correct[gt_predicate] += 1
                self.total[gt_predicate] += 1
            else:
                self.total[gt_predicate] += 1

    def get_recall(self):
        # Avoid divide by zero
        for i in range(self.num_classes):
            if self.total[i] == 0:
                self.total[i] = 1 
        # Skip the 0 class
        # print performance for each class
        per_class_performance = []
        for i in range(1, self.num_classes):
            print('class {} recall: {}'.format(i, self.correct[i] / self.total[i]))
            per_class_performance.append(self.correct[i] / self.total[i])
        # print average performance of each independe class, skip the 0 class
        print('average recall: {}'.format(np.mean(per_class_performance)))
