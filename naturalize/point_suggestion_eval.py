import six
import sys
import os
import json
import numpy as np
import pickle
import os.path
import traceback

import naturalize.identifier_splitter as sp
from naturalize.abstract_renaming_model import AbstractRenamingModel


class PointSuggestionEvaluator:
    def __init__(self):
        self.confidence_threshold = [0, 0.001, 0.005, 0.01, 0.02, 0.04, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75,
                                     0.8, 0.85, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9995, 0.9999, 1]
        self.rank_to_eval = [1, 5]
        self.num_points = 0
        self.num_made_suggestions = np.array([[0] * len(self.confidence_threshold)] * len(self.rank_to_eval))
        self.num_correct_suggestions = np.array([[0] * len(self.confidence_threshold)] * len(self.rank_to_eval))
        self.sum_precisions_suggestions = np.array([[0.] * len(self.confidence_threshold)] * len(self.rank_to_eval))
        self.sum_recalls_suggestions = np.array([[0.] * len(self.confidence_threshold)] * len(self.rank_to_eval))
        self.sum_f1_suggestions = np.array([[0.] * len(self.confidence_threshold)] * len(self.rank_to_eval))

    def add_result(self, confidence, is_correct, is_unk, precision_recall, unk_no_suggest=False):
        """
        Add a single point suggestion as a result.
        """
        confidence = np.array(confidence)
        is_correct = np.array(is_correct, dtype=np.bool)
        is_unk = np.array(is_unk, dtype=np.bool)
        self.num_points += 1
        if is_unk[0] and unk_no_suggest:
            return  # No suggestions
        for i in range(len(self.confidence_threshold)):
            num_confident_suggestions = confidence[confidence >= self.confidence_threshold[i]].shape[0]
            for j in range(len(self.rank_to_eval)):
                rank = self.rank_to_eval[j]
                n_suggestions = min(rank, num_confident_suggestions)

                unk_at_rank = np.where(is_unk[:n_suggestions])[0]
                if unk_at_rank.shape[0] == 0 or not unk_no_suggest:
                    unk_at_rank = n_suggestions + 1  # Beyond our current number of sugestions
                else:
                    unk_at_rank = unk_at_rank[0]

                if min(n_suggestions, unk_at_rank) > 0:
                    self.num_made_suggestions[j][i] += 1
                    if np.any(is_correct[:min(n_suggestions, unk_at_rank)]):
                        self.num_correct_suggestions[j][i] += 1

                    pr, re, f1 = self.get_best_f1(precision_recall[:min(n_suggestions, unk_at_rank)])
                    self.sum_precisions_suggestions[j][i] += pr
                    self.sum_recalls_suggestions[j][i] += re
                    self.sum_f1_suggestions[j][i] += f1

    def get_best_f1(self, suggestions_pr_re_f1):
        """
        Get the "best" precision, recall and f1 score from a list of tuples,
        picking the ones with the best f1
        """
        max_f1 = 0
        max_pr = 0
        max_re = 0
        for suggestion in suggestions_pr_re_f1:
            if suggestion[2] > max_f1:
                max_pr, max_re, max_f1 = suggestion
        return max_pr, max_re, max_f1

    def __str__(self):
        n_made_suggestions = np.array(self.num_made_suggestions, dtype=float)
        n_correct_suggestions = np.array(self.num_correct_suggestions, dtype=float)
        result_string = ""
        for i in range(len(self.rank_to_eval)):
            result_string += "At Rank " + str(self.rank_to_eval[i]) + os.linesep
            result_string += "Suggestion Frequency " + str(
                n_made_suggestions[i] / self.num_points) + os.linesep
            result_string += "Suggestion Exact Match " + str(
                np.divide(n_correct_suggestions[i], n_made_suggestions[i])) + os.linesep
            result_string += "Exact Match@100%: " + str(float(n_correct_suggestions[i, 0]) / self.num_points) + os.linesep

            result_string += "Suggestion Precision " + str(
                np.divide(self.sum_precisions_suggestions[i], n_made_suggestions[i])) + os.linesep
            result_string += "Suggestion Recall " + str(
                np.divide(self.sum_recalls_suggestions[i], n_made_suggestions[i])) + os.linesep
            result_string += "Suggestion F1 " + str(
                np.divide(self.sum_f1_suggestions[i], n_made_suggestions[i])) + os.linesep
            result_string += "Num Points: " + str(self.num_points) + os.linesep
            result_string += "F1@100%: " + str(float(self.sum_f1_suggestions[i, 0]) / self.num_points) + os.linesep
        return result_string


def token_precision_recall(predicted_parts, gold_set_parts):
    """
    Get the precision/recall for the given token.

    :param predicted_parts: a list of predicted parts
    :param gold_set_parts: a list of the golden parts
    :return: precision, recall, f1 as floats
    """
    ground = [tok.lower() for tok in gold_set_parts]
    prediction = list(predicted_parts)

    tp = 0
    for subtoken in prediction:
        if subtoken == "***" or subtoken is None:
            continue  # Ignore UNKs
        if subtoken.lower() in ground:
            ground.remove(subtoken.lower())
            tp += 1

    precision = float(tp) / len(predicted_parts)
    recall = float(tp) / len(gold_set_parts)
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.
    return precision, recall, f1


def evaluate_renamer(evaluated_renamer, evaluation_file):
    evaluation_data = PointSuggestionEvaluator()
    with open(evaluation_file) as f:
        eval_data = json.load(f)
    for document in eval_data:
        code_tokens = document["codeTokens"]
        bound_variables = document["boundVariables"]
        bound_variable_features = document["boundVariableFeatures"]

        for bound_var in range(len(bound_variables)):
            try:
                actual_name = code_tokens[bound_variables[bound_var][0]]
                tokens_copy = list(code_tokens)
                for pos in bound_variables[bound_var]:
                    assert code_tokens[pos] == actual_name
                    tokens_copy[pos] = None  # Make sure that the evaluated model gets no information

                eval_json = [dict(codeTokens=tokens_copy, boundVariables=[bound_variables[bound_var]],
                                  boundVariableFeatures=[bound_variable_features[bound_var]])]

                prediction = evaluated_renamer.predict(eval_json)[0]
                prediction_confidence = [res[1] for res in prediction]
                prediction_is_correct = [res[0] == actual_name for res in prediction]
                prediction_is_unk = [res[0] is None for res in prediction]

                # Always split the identifers independently of what the renamer knows.
                actual_name_parts = sp.split_identifier_into_parts(actual_name)
                prediction_pr = [token_precision_recall(sp.split_identifier_into_parts(res[0]),
                                                        actual_name_parts) for res in prediction]

                evaluation_data.add_result(prediction_confidence, prediction_is_correct, prediction_is_unk,
                                              prediction_pr)
            except Exception:
                print("Failed for identifier %s because %s" % (actual_name, traceback.print_exc()))
    return evaluation_data


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print('Usage <renamer_file> <eval_file>')
        sys.exit(-1)

    renamer_file = sys.argv[1]
    eval_file = sys.argv[2]
    renamer = AbstractRenamingModel.load(renamer_file)

    evaluation_metrics = evaluate_renamer(renamer, eval_file)

    with open('point_eval_no_unk_' + os.path.split(renamer_file)[1][:-4] + '_' + os.path.split(eval_file)[1][:-5] + '.pkl', 'w') as f:
        pickle.dump(evaluation_metrics, f)
    print(evaluation_metrics)

