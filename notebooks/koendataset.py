from torchtext.datasets import TranslationDataset
from pathlib import Path

class KoEn(TranslationDataset):
    urls = []
    name = 'koen'
    dirname = ''

    @classmethod
    def splits(cls, exts, fields, root='./data/translation',
               train='train', validation='dev', test='test', **kwargs):
        """Create dataset objects for splits of the KO-EN translation dataset.
        Arguments:
            exts: A tuple containing the extensions for each language. Must be
                either ('.ko', '.en') or the reverse.
            fields: A tuple containing the fields that will be used for data
                in each language.
            root: Root dataset storage directory. Default is '.data'.
            train: The prefix of the train data. 
            validation: The prefix of the validation data. 
            test: The prefix of the test data. 
            Remaining keyword arguments: Passed to the splits method of
                Dataset.
        """
        if 'path' not in kwargs:
            expected_folder = Path(root).joinpath(cls.name)
            path = str(expected_folder) if expected_folder.exists() else None
        else:
            path = kwargs['path']
            del kwargs['path']

        return super(KoEn, cls).splits(
            exts, fields, path, root, train, validation, test, **kwargs)