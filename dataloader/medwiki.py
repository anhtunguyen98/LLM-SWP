import os
import re
from pathlib import Path

import datasets


logger = datasets.logging.get_logger(__name__)

TXT_PATTERN = r"^.*\.txt$"

_CITATION = """\
@misc{https://doi.org/10.48550/arxiv.2210.06345,
  doi = {10.48550/ARXIV.2210.06345},
  
  url = {https://arxiv.org/abs/2210.06345},
  
  author = {Liévin, Valentin and Motzfeldt, Andreas Geert and Jensen, Ida Riis and Winther, Ole},
  
  keywords = {Computation and Language (cs.CL), Information Retrieval (cs.IR), Machine Learning (cs.LG), FOS: Computer and information sciences, FOS: Computer and information sciences, I.2.7; H.3.3; I.2.1},
  
  title = {Variational Open-Domain Question Answering},
  
  publisher = {arXiv},
  
  year = {2022},
  
  copyright = {arXiv.org perpetual, non-exclusive license}
}
"""

_VERSION = "0.0.1"
_HOMEPAGE = "https://github.com/VodLM"

_DESCRIPTION = """\
The MedWiki corpus, which is distributed under the MIT license, 
consists of a subset of 4.5% of the English Wikipedia articles 
and has been specifically curated for the MedMCQA and USMLE datasets. 
The collection was created by utilizing the Wikipedia API to search 
for articles related to each answer option present in the MedMCQA and 
USMLE datasets. The top ten Wikipedia articles for each answer option 
were selected and included in the final corpus. This subset covers a 
wide range of topics in the field of medicine that could be relevant 
for answering questions in this domain.
questions.
"""

_URL = "https://f001.backblazeb2.com/file/FindZebraData/fz-openqa/datasets/medwiki_v6.zip"


class MedWikipediaCorpusConfig(datasets.BuilderConfig):
    """BuilderConfig for the MedQa English Corpus objecxt."""

    def __init__(self, **kwargs):
        """BuilderConfig for the Corpus object.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(MedWikipediaCorpusConfig, self).__init__(**kwargs)

class MedWikipediaCorpusGenerator(datasets.GeneratorBasedBuilder):
    """MedWikipediaCorpus Dataset. Version 0.0.1"""

    BUILDER_CONFIGS = [
        MedWikipediaCorpusConfig(
            name="plain_text",
            version=datasets.Version(_VERSION, ""),
            description="Plain text",
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "document.idx": datasets.Value("int32"),
                    "document.text": datasets.Value("string"),
                    "document.title": datasets.Value("string"),
                }
            ),
            supervised_keys=None,
            homepage=_HOMEPAGE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        downloaded_file = dl_manager.download_and_extract(_URL)
        if not Path(downloaded_file).is_dir():
            raise Exception(
                f"Could not download the dataset Content of `downloaded_file`:"
                f"{open(downloaded_file, 'r').read()}"
            )
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"output_dir": Path(downloaded_file)},
            )
        ]

    def _generate_examples(self, output_dir: str):
        logger.info("generating examples from = %s", output_dir)
        paths = [
            p
            for p in Path(output_dir).iterdir()
            if p.is_dir() and (p.name.startswith("med_x_wiki") or p.name.startswith("wikipedia"))
        ]
        assert len(paths) == 1, f"Found {len(paths)} directories in {output_dir}: {paths}"
        path = paths[0]

        # list files
        data_files = [os.path.join(path, p) for p in os.listdir(path) if re.findall(TXT_PATTERN, p)]

        # iterate and yield documents
        for i, fn in enumerate(data_files):
            with open(fn, "r") as f:
                # the first line is the title
                title = f.readline()
                text = f.read()
                yield i, {"document.text": text, "document.idx": i, "document.title": title}