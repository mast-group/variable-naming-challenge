import six
import json
import sys

from naturalize.abstract_renaming_model import AbstractRenamingModel
from naturalize.point_suggestion_eval import token_precision_recall

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print('Usage <renamer_file> <eval_file> all|confidenterrors')
        sys.exit(-1)

    renamer_file = sys.argv[1]
    eval_file = sys.argv[2]
    printtype = sys.argv[3]

    renamer = AbstractRenamingModel.load(renamer_file)

    with open(eval_file) as f:
        eval_data = json.load(f)

    for document in eval_data:
        code_tokens = document["codeTokens"]
        bound_variables = document["boundVariables"]
        bound_variable_features = document["boundVariableFeatures"]
        provenance = document["provenance"]
        if printtype == "all":
                print("*" * 120)
                print("For file %s " % provenance)
                print("*" * 120)

        for bound_var in range(len(bound_variables)):
            actual_name = code_tokens[bound_variables[bound_var][0]]
            tokens_copy = list(code_tokens)
            for pos in bound_variables[bound_var]:
                assert code_tokens[pos] == actual_name
                tokens_copy[pos] = None  # Make sure that the evaluated model gets no information

            eval_json = [dict(codeTokens=tokens_copy, boundVariables=[bound_variables[bound_var]],
                              boundVariableFeatures=[bound_variable_features[bound_var]])]

            prediction = renamer.predict(eval_json)[0]

            if printtype == "all":
                print("Actual:" + actual_name)
                print("Predicted:" + str(prediction[:5]))
            elif printtype == "confidenterrors":
                actual_name_parts = renamer.split_identifier(actual_name)
                predicted_name_parts = renamer.split_identifier(prediction[0][0])
                pr, re, f1 = token_precision_recall(predicted_name_parts, actual_name_parts)
                if f1 < .01 and prediction[0][1] > .95 and prediction[0][0] is not None:
                    print("")
                    print("For file %s " % provenance)
                    print("Actual:" + actual_name)
                    print("Predicted:" + str(prediction[:5]))
                else:
                    sys.stdout.write('.')
            else:
                raise Exception("Unrecognized option %s" % printtype)

