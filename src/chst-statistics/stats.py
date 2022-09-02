import imp
import json
import os
from collections import Counter
from pathlib import Path
from turtle import width
from typing import Union

import arabic_reshaper
import matplotlib.pyplot as plt
import numpy as np
from bidi.algorithm import get_display
from hazm import Normalizer, word_tokenize
from loguru import logger
from PIL import Image
from src.data import DATA_DIR
from wordcloud import WordCloud


class ChatStatistics:
    """Generating chat statistics from a telegram json chat file
    """

    def __init__(self, chat_json: Union[str, Path]):
        """
        Args:
            chat_json (Union[str, Path]): path to json file
        """

        # load chat_data
        logger.info(f"reading json file from path ->{chat_json}")
        with open(chat_json) as f:
            self.chat_data = json.load(f)

        self.normalizer = Normalizer()

        # load stopwords
        logger.info(f"reading stopwords form path -> {str(DATA_DIR / 'stopwords.txt')}")
        stop_words = open(DATA_DIR / "stopwords.txt").readlines()
        stop_words = list(map(str.strip, stop_words))
        self.stop_words = list(map(self.normalizer.normalize, stop_words))

    def generate_wordcloud(self, output_dir: Union[str, Path]):
        """Generating wordcloud using chat statistics

        Args:
            output_dir (Union[str, Path]): output directory to save wordcloud file
        """

        logger.info("loading text contect.....")
        text_content = ""

        for msg in self.chat_data["messages"]:
            if type(msg["text"]) is str:
                tokens = word_tokenize(msg["text"])
                tokens = list(filter(lambda item: item not in self.stop_words, tokens))
                text_content += f" {' '.join(tokens)}"

        # Counter(word_tokenize(text_content)).most_common()[:30]

        # generating wordcloud

        logger.info("generating wordcloud.......")
        text_content = self.normalizer.normalize(text_content)
        wordcloud = WordCloud(
            font_path=str(DATA_DIR / "BHoma.ttf"),
            background_color="white",
            height=1200,
            width=1200,
        ).generate(text_content)

        wordcloud.to_file(str(Path(output_dir) / "wordcloud.png"))


if __name__ == "__main__":
    chat_stats = ChatStatistics(chat_json=DATA_DIR / "chat-data.json")
    chat_stats.generate_wordcloud(DATA_DIR)
