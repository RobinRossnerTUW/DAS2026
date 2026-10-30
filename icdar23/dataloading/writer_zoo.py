import os.path
from .regex import ImageFolder

class WriterZoo:

    @staticmethod
    def new(desc, **kwargs):
        return ImageFolder(desc['path'], regex=desc['regex'], **kwargs)

    @staticmethod
    def get(dataset, set, **kwargs):
        _all = WriterZoo.datasets
        d = _all[dataset]
        s = d['set'][set]

        s['path'] = os.path.join(d['basepath'], s['path'])
        return WriterZoo.new(s, **kwargs)

    datasets = {

        'icdar2017': {
            'basepath': '/data/rrossner/datasets/ICDAR17',
            'set': {
                'test': {
                    'path': '/data/rrossner/DAS2026/icdar/test_patches_coords',
                    'regex': {'writer': '(\d+)', 'page': '\d+-IMG_MAX_(\d+)'}
                },
                'train': {
                    'path': '/data/rrossner/DAS2026/icdar/train_patches',
                    'regex': {
                        'cluster': '(\d+)',
                        'writer':  '\d+_(\d+)',
                        'page':    '\d+_\d+-IMG_MAX_(\d+)',
                    }
                },
            }
        },

        'icdar2013': {
            'basepath': '/data/mpeer/resources',
            'set': {
                'test' :  {'path': 'icdar2013_test_sift_patches_binarized',
                                  'regex' : {'writer': '(\d+)', 'page': '\d+_(\d+)'}},

                'train' :  {'path': 'icdar2013_train_sift_patches_1000/',
                                  'regex' : {'cluster' : '(\d+)', 'writer': '\d+_(\d+)', 'page' : '\d+_\d+_(\d+)'}}                             
            }
        },
        
        'icdar2019': {
            'basepath': '/data/mpeer/resources',
            'set': {
                'test' :  {'path': 'wi_comp_19_test_patches',
                                  'regex' : {'writer': '(\d+)', 'page': '\d+_(\d+)'}},

                'train' :  {'path': 'wi_comp_19_validation_patches',
                                  'regex' : {'cluster' : '(\d+)', 'writer': '\d+_(\d+)', 'page' : '\d+_\d+_(\d+)'}},
            }
        },

        # ── CVL Database ─────────────────────────────────────────────────────
        # Patch filenames: 0052-1-cropped_0.png
        #   writer = 0052  (digits before first dash)
        #   page   = 1     (text number after first dash)
        # No cluster folders — train uses writer labels directly.
        # Paths are PLACEHOLDERS — patched at runtime by the sbatch script.
        'cvl': {
            'basepath': '',
            'set': {
                'test': {
                    'path': '/data/rrossner/datasets/CVL/cvl_test_patches_bin',
                    'regex': {
                        'writer': '(\d+)-\d+-cropped',
                        'page':   '\d+-(\d+)-cropped',
                    }
                },
                'train': {
                    'path': '/data/rrossner/datasets/CVL/cvl_train_patches_bin',
                    'regex': {
                        'writer': '(\d+)-\d+-cropped',
                        'page':   '\d+-(\d+)-cropped',
                    }
                },
            }
        },

        # ── MASLOTS Sanskrit hands (Patrick-data) ────────────────────────────
        # Patch filenames: 0_0012_345.png  (writer_page_patch)
        # writer = hand id, see /data/rrossner/datasets/Patrick-data/labels.csv
        'maslots_skt': {
            'basepath': '',
            'set': {
                'test': {
                    'path': '/data/rrossner/datasets/Patrick-data_patches',
                    'regex': {'writer': '(\d+)_', 'page': '\d+_(\d+)'}
                },
            }
        },
    }
