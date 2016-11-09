import six
import pickle


class AbstractRenamingModel:
    """
    An abstract renaming model, to be used for all renaming models.

    The input JSON has the following format:
    [file_binding1, file_binding2, ...]

    each file_binding has the following format:
    {   "codeTokens": ["list", "of", "all", "code", "tokens", ...],
        "boundVariables": [[10, 20, ...], [40, 63, 102, ...], ... ]
        "boundVariableFeatures": [["final", "static", ...], ...]
    }

    i.e. codeTokens contains all tokens of some file;
         boundVariables contain a list of lists of all the indexes in codeTokens that are bound together;
         boundVariableFeatures contains a list of lists of all the (string) features for each bound token in
                boundVariables.

        Indexes of boundVariables and boundVariableFeatures match i.e. boundVariables[i] and boundVariableFeatures[i]
            refer to the same binding.
    """

    def train(self, input_json):
        """
        Ask the model to train itself using the input_json JSON file.

        :param input_json: The parsed .json object
        """
        raise NotImplementedError()

    def predict(self, input_json):
        """
        Given an input json, predict the name of the variables. For each of the boundVariables a list tuples will be
        returned. ie.

        [var_suggestion1, var_suggestion2, ...]

        where

        var_suggestion1 is an *ordered* list of suggestions, along with their confidence score [0,1]
        [ ('varName', score), ...]

        when 'varName' is None, then an UNK (unknown) variable name is assumed.


        :param input_json: The input JSON with exactly the same format as described above. However, for the
        suggestion points, it may contain anything (and possibly None).
        :return:
        """
        raise NotImplementedError()

    def save(self, filename):
        """
        Save this renaming model to a file.
        :param filename:
        :return:
        """
        raise NotImplementedError()

    def split_identifier(self, identifier):
        """
        Split identifier into smaller parts that the renamer may be able to predict. By default no splitting will be
        done, but implementers may choose to override this.

        :param identifier: the identifier to be split
        :type identifier: str
        :return:
        """
        return [identifier]

    @staticmethod
    def load(filename):
        """
        Load a pickled file.
        :param filename: the input filename.
        :rtype: AbstractRenamingModel
        """
        with open(filename, 'rb') as f:
            return pickle.load(f)
