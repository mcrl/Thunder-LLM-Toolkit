# Copyright 2023 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Library of instructions."""

import collections
import json
import logging
import random
import re
import string
import unicodedata
from typing import Dict, Optional, Sequence, Union

import langdetect

import importlib
instructions_util = importlib.import_module("thunderllm.inference.tasks.thunderllm-benchmark.thunderllm-snu-ko-ifeval.instructions_util")


logger = logging.getLogger(__name__)

_InstructionArgsDtype = Optional[Dict[str, Union[int, str, Sequence[str]]]]

_LANGUAGES = instructions_util.LANGUAGE_CODES

# The relational operation for comparison.
_COMPARISON_RELATION = ("미만", "이상")

# The maximum number of sentences.
_MAX_NUM_SENTENCES = 20

# The number of placeholders.
_NUM_PLACEHOLDERS = 4

# The number of bullet lists.
_NUM_BULLETS = 5

# The options of constrained response.
_CONSTRAINED_RESPONSE_OPTIONS = (
    "네",
    "아니요",
    "아마도요",
)

# The options of starter keywords.
_STARTER_OPTIONS = (
)

# The options of ending keywords.
_ENDING_OPTIONS = ("다른 궁금한 점 있으신가요?", "더 도와드릴 부분이 있을까요?")

# The number of highlighted sections.
_NUM_HIGHLIGHTED_SECTIONS = 4

# The section splitter.
_SECTION_SPLITER = ("섹션", "단락")

# The number of sections.
_NUM_SECTIONS = 5

# The number of paragraphs.
_NUM_PARAGRAPHS = 5

# The postscript marker.
_POSTSCRIPT_MARKER = ("P.S.", "추신:")

# The number of keywords.
_NUM_KEYWORDS = 2 

# The occurrences of a single keyword.
_KEYWORD_FREQUENCY = 3

# The occurrences of a single letter.
_LETTER_FREQUENCY = 10

# The occurrences of words with all capital letters.
_ALL_CAPITAL_WORD_FREQUENCY = 20

# The number of words in the response.
_NUM_WORDS_LOWER_LIMIT = 100
_NUM_WORDS_UPPER_LIMIT = 500

# The number of Korean letters in the response.
_NUM_LETTERS_LOWER_LIMIT = 100
_NUM_LETTERS_UPPER_LIMIT = 500

# Multiple options
_MULTIPLE_OPTIONS = ("A", "B", "C", "D")


class Instruction:
    """An instruction template."""

    def __init__(self, instruction_id):
        self.id = instruction_id

    def build_description(self, **kwargs):
        raise NotImplementedError("`build_description` not implemented.")

    def get_instruction_args(self):
        raise NotImplementedError("`get_instruction_args` not implemented.")

    def get_instruction_args_keys(self):
        raise NotImplementedError("`get_instruction_args_keys` not implemented.")

    def check_following(self, value):
        raise NotImplementedError("`check_following` not implemented.")


class ResponseLanguageChecker(Instruction):
    """Check the language of the entire response."""

    def build_description(self, *, language=None):
        """Build the instruction description.

        Args:
          language: A string representing the expected language of the response.
          en: english, ko: korean

        Returns:
          A string representing the instruction description.
        """
        self._language = language
        if self._language is None:
            self._language = random.choice(list(_LANGUAGES.keys()))

        self._description_pattern = (
            "답변은 {language}로 작성해야 하며 다른 언어는 허용되지 않습니다."
        )
        return self._description_pattern.format(language=_LANGUAGES[self._language])

    def get_instruction_args(self):
        """Returns the keyword args of `build_description`."""
        return {"language": self._language}

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["language"]

    def check_following(self, value):
        """Check if the language of the entire response follows the instruction.

        Args:
          value: A string representing the response.

        Returns:
          True if the language of `value` follows instruction; otherwise False.
        """
        assert isinstance(value, str)

        try:
            return langdetect.detect(value) == self._language
        except langdetect.LangDetectException as e:
            # Count as instruction is followed.
            logging.error(
                "Unable to detect language for text %s due to %s", value, e
            )  # refex: disable=pytotw.037
            return True


class NumberOfSentences(Instruction):
    """Check the number of sentences. No need to modify, because we don't have this instruction in ko ifeval"""

    def build_description(self, *, num_sentences=None, relation=None):
        """Build the instruction description.

        Args:
          num_sentences: An integer specifying the number of sentences as a
            threshold.
          relation: A string in (`less than`, `at least`), defining the relational
            operator for comparison.
            Two relational comparisons are supported for now:
            if 'less than', the actual number of sentences < the threshold;
            if 'at least', the actual number of sentences >= the threshold.

        Returns:
          A string representing the instruction description.
        """
        # The number of sentences as a threshold for comparison.
        self._num_sentences_threshold = num_sentences
        if self._num_sentences_threshold is None or self._num_sentences_threshold < 0:
            self._num_sentences_threshold = random.randint(1, _MAX_NUM_SENTENCES)

        if relation is None:
            self._comparison_relation = random.choice(_COMPARISON_RELATION)
        elif relation not in _COMPARISON_RELATION:
            raise ValueError(
                "The supported relation for comparison must be in "
                f"{_COMPARISON_RELATION}, but {relation} is given."
            )
        else:
            self._comparison_relation = relation

        self._description_pattern = (
            "Your response should contain {relation} {num_sentences} sentences."
        )
        return self._description_pattern.format(
            relation=self._comparison_relation,
            num_sentences=self._num_sentences_threshold,
        )

    def get_instruction_args(self):
        """Returns the keyword args of `build_description`."""
        return {
            "num_sentences": self._num_sentences_threshold,
            "relation": self._comparison_relation,
        }

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["num_sentences", "relation"]

    def check_following(self, value):
        """Check if the number of sentences follows the instruction.

        Args:
          value: A string representing the response.

        Returns:
          True if the response follows the instruction.

        Raise:
            ValueError if the string in `instruction_args` is not in
            [`less_than`, `at_least`].
        """
        num_sentences = instructions_util.count_sentences(value)
        if self._comparison_relation == _COMPARISON_RELATION[0]:
            return num_sentences < self._num_sentences_threshold
        elif self._comparison_relation == _COMPARISON_RELATION[1]:
            return num_sentences >= self._num_sentences_threshold


class PlaceholderChecker(Instruction):
    """Check the placeholders in template writing."""

    def build_description(self, *, num_placeholders=None):
        """Build the instruction description.

        Args:
          num_placeholders: An integer denoting the minimum number of
            placeholders required in the response.

        Returns:
          A string representing the instruction description.
        """
        self._num_placeholders = num_placeholders
        if self._num_placeholders is None or self._num_placeholders < 0:
            self._num_placeholders = random.randint(1, _NUM_PLACEHOLDERS)
        self._description_pattern = (
            "[주소]와 같이 대괄호로 표시된 플레이스 홀더를 {num_placeholders}개 이상 포함해야 합니다."
        )
        return self._description_pattern.format(num_placeholders=self._num_placeholders)

    def get_instruction_args(self):
        """Returns the keyword args of `build_description`."""
        return {"num_placeholders": self._num_placeholders}

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["num_placeholders"]

    def check_following(self, value):
        """Check if the number of placeholders follows the instruction.

        Args:
          value: A string representing the response.

        Returns:
          True if the actual number of placeholders in the response is greater than
          or equal to `num_placeholders`; otherwise, False.
        """
        placeholders = re.findall(r"\[.*?\]", value)
        num_placeholders = len(placeholders)
        return num_placeholders >= self._num_placeholders


class BulletListChecker(Instruction):
    """Checks the bullet list in the response."""

    def build_description(self, *, num_bullets=None):
        """Build the instruction description.

        Args:
          num_bullets: An integer specifying the exact number of bullet points
            that should appear in the response.

        Returns:
          A string representing the instruction description.
        """
        # Set the number of bullets; if not provided, use a random number
        self._num_bullets = num_bullets
        if self._num_bullets is None or self._num_bullets < 0:
            self._num_bullets = random.randint(1, _NUM_BULLETS)
        self._description_pattern = (
            "답변에서는 정확히 {num_bullets}개의 글머리 기호를 포함해야 합니다.\n"
            + "다음과 같이 마크다운 형식으로 글머리 기호를 사용하세요:\n"
            + "- 글머리 기호 1\n"
            + "- 글머리 기호 2\n"
            + "- 글머리 기호 3"
        )
        return self._description_pattern.format(num_bullets=self._num_bullets)

    def get_instruction_args(self):
        """Returns the keyword args of `build_description`."""
        return {"num_bullets": self._num_bullets}

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["num_bullets"]

    def check_following(self, value):
        r"""Check if the number of bullet points meets the requirement.

        Args:
          value: A string representing the response. The response is expected to
            contain bullet points starting with `-`.

        Returns:
          True if the actual number of bullet points in the response meets the
          requirement, otherwise False.
        """
        # Match only bullets that start with `-`
        bullet_lists = re.findall(r"^\s*-\s.*$", value, flags=re.MULTILINE)
        num_bullet_lists = len(bullet_lists)
        return num_bullet_lists == self._num_bullets


class ConstrainedResponseChecker(Instruction):
    """Checks the constrained response."""

    def build_description(self):
        """Build the instruction description."""
        # A sequence of string(s) representing the options of the expected response.
        self._constrained_responses = _CONSTRAINED_RESPONSE_OPTIONS
        self._description_pattern = (
            "다음의 선택지 중 하나로 대답하세요. : {response_options}"
        )
        return self._description_pattern.format(
            response_options=self._constrained_responses
        )

    def get_instruction_args(self):
        """Returns the keyword args of `build_description`."""
        return None

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return []

    def check_following(self, value):
        """Checks if the response matches the constrained options.

        Args:
          value: A string representing the response.

        Returns:
          True if the actual response contains one of the options in the constrained
          responses; otherwise False.
        """
        result = False
        for response in self._constrained_responses:
            if response in value:
                filtered_value = re.sub(r"[^\w\s]", "", value.strip())
                if filtered_value in self._constrained_responses:
                    result = True
                    break
        return result


class HighlightSectionChecker(Instruction):
    """Checks the highlighted section."""

    def build_description(self, *, num_highlights=None):
        """Build the instruction description.

        Args:
          num_highlights: An integer specifying the minimum number of highlighted
            sections.

        Returns:
          A string representing the instruction description.
        """
        self._num_highlights = num_highlights
        if self._num_highlights is None or self._num_highlights < 0:
            self._num_highlights = random.randint(1, _NUM_HIGHLIGHTED_SECTIONS)

        self._description_pattern = (
            "마크다운으로 답변에서 {num_highlights}개 이상의 섹션 제목을 강조하세요."
            + "(예: *섹션 제목* 또는 **섹션 제목**)"
        )

        return self._description_pattern.format(num_highlights=self._num_highlights)

    def get_instruction_args(self):
        """Returns the keyword args of `build_description`."""
        return {"num_highlights": self._num_highlights}

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["num_highlights"]

    def check_following(self, value):
        """Checks if the number of highlighted sections meets the requirement.

        Args:
          value: a string representing the response. The response is expected to
            contain highlighted sections in the format of *highlighted* or **highlighted**.

        Returns:
          True if the actual number of highlighted sections in the format of
          *highlighted sections* or **highlighted sections** meets the minimum requirement; 
          otherwise False.
        """
        num_highlights = 0
        highlights = re.findall(r"(?<!\S)\*{1,2}(.*?)\*{1,2}(?!\S)", value, re.DOTALL)
        for highlight in highlights:
            if highlight.strip("*").strip():
                num_highlights += 1

        return num_highlights >= self._num_highlights


class SectionChecker(Instruction):
    """Checks the sections."""

    def build_description(self, *, section_spliter=None, num_sections=None):
        """Build the instruction description.

        Args:
          section_spliter: A string represents the section spliter keyword that
            marks a new section, i.e., `섹션` or `단락`.
          num_sections: An integer specifying the number of sections.

        Returns:
          A string representing the instruction description.
        """
        self._section_spliter = (
            section_spliter.strip()
            if isinstance(section_spliter, str)
            else section_spliter
        )
        if self._section_spliter is None:
            self._section_spliter = random.choice(_SECTION_SPLITER)

        self._num_sections = num_sections
        if self._num_sections is None or self._num_sections < 0:
            self._num_sections = random.randint(1, _NUM_SECTIONS)

        section_description = (
            "답변을 {num_sections}개의 섹션으로 나누어 쓰세요. "
            + "각 섹션의 시작 부분을 {section_spliter} X로 표시하세요. 예를 들어:\n" 
            + "{section_spliter} 1\n" 
            + "[섹션 1의 내용]\n" 
            + "{section_spliter} 2\n" 
            + "[섹션 2의 내용]"
        )
        date_description = (
            "답변을 {num_sections}개의 섹션으로 나누어 쓰세요. "
            + "각 섹션의 시작 부분을 X{section_spliter}로 표시하세요. 예를 들어:\n" 
            + "1{section_spliter}\n" 
            + "[1일차의 내용]\n" 
            + "2{section_spliter}\n" 
            + "[2일차의 내용]"
        )

        if self._section_spliter == "일차":
            self._description_pattern = (date_description)
        else:
            self._description_pattern = (section_description)

        return self._description_pattern.format(
            num_sections=self._num_sections, section_spliter=self._section_spliter
        )

    def get_instruction_args(self):
        """Returns the keyword args of `build_description`."""
        return {
            "section_spliter": self._section_spliter,
            "num_sections": self._num_sections,
        }

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["section_spliter", "num_sections"]

    def check_following(self, value):
        """Checks the response contains multiple sections.

        Args:
          value: A string representing the response. The response is expected
            to contain multiple sections (number of sections is greater than 1).
            A new section starts with `섹션 1 / 1일차`, where the number denotes the
            section index.

        Returns:
          True if the number of sections in the response is greater than or equal to
          the minimum number of sections; otherwise, False.
        """
        if self._section_spliter == "일차":
            section_splitter_patten = r"\d+일차"
        else:
            section_splitter_patten = r"\s?" + self._section_spliter + r"\s?\d+\s?"
        sections = re.split(section_splitter_patten, value)
        num_sections = len(sections) - 1
        return num_sections >= self._num_sections


class ParagraphChecker(Instruction):
    """Checks the paragraphs."""

    def build_description(self, *, num_paragraphs=None):
        """Build the instruction description.

        Args:
          num_paragraphs: An integer specifying the number of paragraphs.

        Returns:
          A string representing the instruction description.
        """
        self._num_paragraphs = num_paragraphs
        if self._num_paragraphs is None or self._num_paragraphs < 0:
            self._num_paragraphs = random.randint(1, _NUM_PARAGRAPHS)

        self._description_pattern = (
            "{num_paragraphs}문단으로 나눠서 쓰세요. "
            + "마크다운 구분선(***/* * *)으로 문단을 구분하세요."
        )

        return self._description_pattern.format(num_paragraphs=self._num_paragraphs)

    def get_instruction_args(self):
        """Returns the keyword args of `build_description`."""
        return {"num_paragraphs": self._num_paragraphs}

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["num_paragraphs"]

    def check_following(self, value):
        """Checks the response contains required number of paragraphs.

        Args:
          value: A string representing the response. The response may contain
            paragraphs that are separated by the markdown divider: `***`or '* * *'.

        Returns:
          True if the actual number of paragraphs is the same as required;
          otherwise, False.
        """
        divider_pattern = r"\s?(\*\*\*|\* \* \*)\s?"

        paragraphs = re.split(divider_pattern, value)
        filtered_paragraphs = [p.strip() for p in paragraphs if p.strip() 
                               and not re.match(divider_pattern, p.strip())]

        return len(filtered_paragraphs) == self._num_paragraphs


class PostscriptChecker(Instruction):
    """Checks the postscript."""

    def build_description(self, *, postscript_marker=None):
        """Build the instruction description.

        Args:
          postscript_marker: A list of strings containing keywords that mark the start
            of the postscript section.

        Returns:
          A string representing the instruction description.
        """
        self._postscript_marker = (
            postscript_marker.strip()
            if isinstance(postscript_marker, str)
            else postscript_marker
        )
        if self._postscript_marker is None:
            self._postscript_marker = random.choice(_POSTSCRIPT_MARKER)

        self._description_pattern = (
            "답변 마지막에 {postscript} 시작하는 추신을 명시적으로 추가하세요."
        )

        return self._description_pattern.format(postscript=self._postscript_marker)

    def get_instruction_args(self):
        """Returns the keyword args of `build_description`."""
        return {"postscript_marker": self._postscript_marker}

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["postscript_marker"]

    def check_following(self, value):
        """Checks if the response follows the postscript format.

        Args:
        value: A string representing the response. The response is expected to
            contain a postscript section.

        Returns:
        True if the response contains a postscript section starting with
        any of the specified markers; otherwise False.
        """
        # Generate regex pattern based on the specified marker
        value = value.lower()
        if self._postscript_marker == "P.S.":
            # Match "P.S." regardless of whitespace
            postscript_pattern = r"(?i)p\.\s*s\..*"
        elif self._postscript_marker == "추신: ":
            # Match "추신:" regardless of whitespace
            postscript_pattern = r"(?i)추신:.*"
        else:
            # Match custom markers
            postscript_pattern = rf"(?i){re.escape(self._postscript_marker)}.*"

        # Check for matches
        postscript = re.findall(postscript_pattern, value, flags=re.MULTILINE)
        return bool(postscript)


class KeywordChecker(Instruction):
    """Check the exisitence of certain keywords."""

    def build_description(self, *, keywords=None):
        """Build the instruction description.

        Args:
          keywords: A sequence of strings representing the keywords that are
            expected in the response.

        Returns:
          A string representing the instruction description.
        """

        if not keywords:
            self._keywords = instructions_util.generate_keywords(
                num_keywords=_NUM_KEYWORDS
            )
        else:
            self._keywords = keywords
        self._keywords = sorted(self._keywords)

        self._description_pattern = "응답에 다음 단어를 포함하세요. : {keywords}"

        return self._description_pattern.format(keywords=self._keywords)

    def get_instruction_args(self):
        """Returns the keyword args of `build_description`."""
        return {"keywords": self._keywords}

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["keywords"]

    def check_following(self, value):
        """Check if the response contain the expected keywords."""
        for keyword in self._keywords:
            if not re.search(keyword, value, flags=re.IGNORECASE):
                return False
        return True


class KeywordFrequencyChecker(Instruction):
    """Check the keyword frequency."""

    def build_description(self, *, keyword=None, frequency=None, relation=None):
        """Build the instruction description.

        Args:
          keyword: A string representing a keyword that is expected in the response.
          frequency: An integer specifying the number of times `keyword` is expected
            to appear in the response.
          relation: A string in (`미만`, `이상`), defining the relational
            operator for comparison.
            Two relational comparisons are supported for now:
            if '미만', the actual number of occurrences < frequency;
            if '이상', the actual number of occurrences >= frequency.

        Returns:
          A string representing the instruction description.
        """
        if not keyword:
            self._keyword = instructions_util.generate_keywords(num_keywords=1)[0]
        else:
            self._keyword = keyword.strip()

        self._frequency = frequency
        if self._frequency is None or self._frequency < 0:
            self._frequency = random.randint(1, _KEYWORD_FREQUENCY)

        if relation is None:
            self._comparison_relation = random.choice(_COMPARISON_RELATION)
        elif relation not in _COMPARISON_RELATION:
            raise ValueError(
                "The supported relation for comparison must be in "
                f"{_COMPARISON_RELATION}, but {relation} is given."
            )
        else:
            self._comparison_relation = relation

        self._description_pattern = (
            "응답에서 다음 단어가 {frequency}번 {relation} 나타나야 합니다. : '{keyword}'"
        )

        return self._description_pattern.format(
            keyword=self._keyword,
            relation=self._comparison_relation,
            frequency=self._frequency,
        )

    def get_instruction_args(self):
        """Returns the keyword args of `build_description`."""
        return {
            "keyword": self._keyword,
            "frequency": self._frequency,
            "relation": self._comparison_relation,
        }

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["keyword", "frequency", "relation"]

    def check_following(self, value):
        """Checks if the response contain the keyword with required frequency."""
        actual_occurrences = len(re.findall(self._keyword, value, flags=re.IGNORECASE))

        if self._comparison_relation == _COMPARISON_RELATION[0]:
            return actual_occurrences < self._frequency
        elif self._comparison_relation == _COMPARISON_RELATION[1]:
            return actual_occurrences >= self._frequency


class NumberOfWords(Instruction):
    """Checks the number of words and ensures the response is in English."""

    def build_description(self, *, num_words=None, relation=None):
        """Build the instruction description.

        Args:
          num_words: An integer specifying the number of words contained in the
            response.
          relation: A string in (`미만`, `이상`), defining the relational
            operator for comparison.
            Two relational comparisons are supported for now:
            if '미만', the actual number of words < num_words;
            if '이상', the actual number of words >= num_words.

        Returns:
          A string representing the instruction description.
        """

        self._num_words = num_words
        if self._num_words is None or self._num_words < 0:
            self._num_words = random.randint(
                _NUM_WORDS_LOWER_LIMIT, _NUM_WORDS_UPPER_LIMIT
            )

        if relation is None:
            self._comparison_relation = random.choice(_COMPARISON_RELATION)
        elif relation not in _COMPARISON_RELATION:
            raise ValueError(
                f"비교를 위한 관계 연산자는 {_COMPARISON_RELATION} 중 하나여야 합니다. "
                f"그러나 {relation}이(가) 제공되었습니다."
            )
        else:
            self._comparison_relation = relation

        self._description_pattern = (
            "답변은 {num_words}단어 {relation}으로 작성하세요. "
            + "반드시 영어로 작성해야 합니다. "
        )

        return self._description_pattern.format(
            relation=self._comparison_relation, num_words=self._num_words
        )

    def get_instruction_args(self):
        """Returns the keyword args of `build_description`."""
        return {"num_words": self._num_words, "relation": self._comparison_relation}

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["num_words", "relation"]

    def check_following(self, value):
        """Check if the text does not contain Korean characters and the response contains the expected number of words."""
        # Check if there are any Korean characters in the value
        has_korean = bool(re.search(r"[ㄱ-ㅎㅏ-ㅣ가-힣]", value))
        if has_korean:
            return False

        # Count the number of words in the value
        num_words = len(value.split())

        if self._comparison_relation == _COMPARISON_RELATION[0]:  # '미만'
            return num_words < self._num_words
        elif self._comparison_relation == _COMPARISON_RELATION[1]:  # '이상'
            return num_words >= self._num_words


class JsonFormat(Instruction):
    """Check the Json format."""

    def build_description(self):
        self._description_pattern = (
            "전체 답변을 JSON 형식으로 구조화하세요. "
            "마크다운 코드블럭(```)을 사용할 수 있습니다."
        )
        return self._description_pattern

    def get_instruction_args(self):
        """Returns the keyword args of `build_description`."""
        return None

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return []

    def check_following(self, value):
        value = (
            value.strip()
            .removeprefix("```json")
            .removeprefix("```Json")
            .removeprefix("```JSON")
            .removeprefix("```")
            .removesuffix("```")
            .strip()
        )
        try:
            json.loads(value)
        except json.JSONDecodeError:
            return False
        except ValueError:
            return False
        return True


class ParagraphFirstWordCheck(Instruction):
    """Check the paragraph and the first word of the nth paragraph."""

    def build_description(
        self, num_paragraphs=None, nth_paragraph=None, first_word=None
    ):
        r"""Build the instruction description.

        Args:
          num_paragraphs: An integer indicating the number of paragraphs expected
            in the response. A paragraph is a subset of the string that is
            expected to be separated by '\n\n'.
          nth_paragraph: An integer indicating the paragraph number that we look at.
            Note that n starts from 1.
          first_word: A string that represent the first word of the bth paragraph.

        Returns:
          A string representing the instruction description.
        """
        self._num_paragraphs = num_paragraphs
        if self._num_paragraphs is None or self._num_paragraphs < 0:
            self._num_paragraphs = random.randint(1, _NUM_PARAGRAPHS)

        self._nth_paragraph = nth_paragraph
        if (
            self._nth_paragraph is None
            or self._nth_paragraph <= 0
            or self._nth_paragraph > self._num_paragraphs
        ):
            self._nth_paragraph = random.randint(1, self._num_paragraphs + 1)

        self._first_word = first_word
        if self._first_word is None:
            self._first_word = instructions_util.generate_keywords(num_keywords=1)[0]
        self._first_word = self._first_word.lower()

        self._description_pattern = (
            "{num_paragraphs}개의 문단으로 답변을 작성하세요. "
            + "각 문단은 두 개의 줄바꿈 문자(\\n)로 구분하세요. "
            + "{nth_paragraph}번째 문단은 {first_word}이라는 단어로 시작해야 합니다."
        )

        return self._description_pattern.format(
            num_paragraphs=self._num_paragraphs,
            nth_paragraph=self._nth_paragraph,
            first_word=self._first_word,
        )

    def get_instruction_args(self):
        """Returns the keyword args of `build_description`."""
        return {
            "num_paragraphs": self._num_paragraphs,
            "nth_paragraph": self._nth_paragraph,
            "first_word": self._first_word,
        }

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["num_paragraphs", "nth_paragraph", "first_word"]

    def check_following(self, value):
        """Checks for required number of paragraphs and correct first word.

        Args:
          value: a string representing the response. The response may contain
            paragraphs that are separated by two new lines and the first word of
            the nth paragraph will have to match a specified word.

        Returns:
          True if the number of paragraphs is the same as required and the first
          word of the specified paragraph is the same as required. Otherwise, false.
        """

        paragraphs = re.split(r"\n\n", value)
        num_paragraphs = len(paragraphs)

        for paragraph in paragraphs:
            if not paragraph.strip():
                num_paragraphs -= 1

        # check that index doesn't go out of bounds
        if self._nth_paragraph <= num_paragraphs:
            paragraph = paragraphs[self._nth_paragraph - 1].strip()
            if not paragraph:
                return False
        else:
            return False

        first_word = ""
        punctuation = {".", ",", "?", "!", "'", '"'}

        # get first word and remove punctuation
        word = paragraph.split()[0].strip()
        # TODO(jeffrey): make more complex?
        word = word.lstrip("'")
        word = word.lstrip('"')

        for letter in word:
            if letter in punctuation:
                break
            first_word += letter.lower()

        return num_paragraphs == self._num_paragraphs and first_word.startswith(self._first_word)


class ForbiddenWords(Instruction):
    """Checks that specified words are not used in response."""

    def build_description(self, *, forbidden_words=None):
        """Build the instruction description.

        Args:
          forbidden_words: A sequences of strings representing words that are not
            allowed in the response.

        Returns:
          A string representing the instruction description.
        """

        if not forbidden_words:
            self._forbidden_words = instructions_util.generate_keywords(
                num_keywords=_NUM_KEYWORDS
            )
        else:
            self._forbidden_words = list(set(forbidden_words))
        self._forbidden_words = sorted(self._forbidden_words)
        self._description_pattern = (
            "{forbidden_words}(이)라는 단어들은 포함하지 마세요."
        )

        return self._description_pattern.format(forbidden_words=self._forbidden_words)

    def get_instruction_args(self):
        """Returns the keyword args of `build_description`."""
        return {"forbidden_words": self._forbidden_words}

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["forbidden_words"]

    def check_following(self, value):
        """Check if the response does not contain the expected keywords."""
        for word in self._forbidden_words:
            if word in value:
                return False
        return True


class TwoResponsesChecker(Instruction):
    """Check that two responses were given."""

    def build_description(self):
        """Build the instruction description."""
        self._description_pattern = (
            "두 가지 다른 응답을 주세요. 별표 기호 6개(******)를 사용하여 응답을 구분하세요."
        )
        return self._description_pattern

    def get_instruction_args(self):
        """Returns the keyword args of `build_description`."""
        return None

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return []

    def check_following(self, value):
        """Checks if the response has two different answers.

        Args:
          value: A string representing the response.

        Returns:
          True if two responses are detected and false otherwise.
        """
        valid_responses = list()
        responses = value.split("******")
        for index, response in enumerate(responses):
            if not response.strip():
                if index != 0 and index != len(responses) - 1:
                    return False
            else:
                valid_responses.append(response)
        return (
            len(valid_responses) == 2
            and valid_responses[0].strip() != valid_responses[1].strip()
        )


class RepeatPromptThenAnswer(Instruction):
    """Checks that Prompt is first repeated then answered."""

    def build_description(self, *, prompt_to_repeat=None):
        """Build the instruction description.

        Args:
          prompt_to_repeat: The prompt that is meant to be repeated.

        Returns:
          A string representing the instruction description.
        """
        if not prompt_to_repeat:
            raise ValueError("prompt_to_repeat must be set.")
        else:
            self._prompt_to_repeat = prompt_to_repeat
            self._description_pattern = (
                "먼저 요청을 변경 없이 반복한 다음 답하세요. "
                "(1. 요청을 반복하기 전에는 어떤 문자도 말하지 말고,"
                " 2. 반복해야 하는 요청에 이 문장은 포함되어 있지 않아야 합니다)"
            )
        return self._description_pattern

    def get_instruction_args(self):
        return {"prompt_to_repeat": self._prompt_to_repeat}

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["prompt_to_repeat"]

    def check_following(self, value):
        # 한글 prompt를 unicode normalize
        prompt_to_repeat_norm = unicodedata.normalize("NFC", self._prompt_to_repeat.strip())
        value_norm = unicodedata.normalize("NFC", value.strip())

        if value_norm.startswith(prompt_to_repeat_norm):
            return True
        return False


class EndChecker(Instruction):
    """Checks that the prompt ends with a given phrase."""

    def build_description(self, *, end_phrase=None):
        """Build the instruction description.

        Args:
          end_phrase: A string representing the phrase the response should end with.

        Returns:
          A string representing the instruction description.
        """
        self._end_phrase = (
            end_phrase.strip() if isinstance(end_phrase, str) else end_phrase
        )
        if self._end_phrase is None:
            self._end_phrase = random.choice(_ENDING_OPTIONS)
        self._description_pattern = (
            "다음과 같은 문구로 전체 답변을 마무리하세요: '{ender}'. "
            "뒤에 다른 단어를 추가해서는 안 됩니다."
        )
        return self._description_pattern.format(ender=self._end_phrase)

    def get_instruction_args(self):
        return {"end_phrase": self._end_phrase}

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["end_phrase"]

    def check_following(self, value):
        """Checks if the response ends with the expected phrase."""
        value = value.strip().strip('"').lower()
        self._end_phrase = self._end_phrase.strip().lower()
        return value.endswith(self._end_phrase)


class TitleChecker(Instruction):
    """Checks the response for a title."""

    def build_description(self):
        """Build the instruction description."""
        self._description_pattern = (
            "답변의 첫 줄에 해시 기호로 시작하는 제목을 추가하세요. (예: # 제목)"
        )
        return self._description_pattern

    def get_instruction_args(self):
        """Returns None as no specific args are needed."""
        return None

    def get_instruction_args_keys(self):
        """Returns an empty list since no args are used."""
        return []

    def check_following(self, value):
        """Checks if the first line of the response contains a title starting with a hash (#)."""
        # Get the first line after stripping leading/trailing whitespace
        first_line = value.strip().splitlines()[0]
        # Matches a hash (#) with optional leading spaces
        pattern = r"^\s*#.+"  
        if re.match(pattern, first_line):
            return True
        return False


class LetterFrequencyChecker(Instruction):
    """Checks letter frequency."""

    def build_description(self, *, letter=None, let_frequency=None, let_relation=None):
        """Build the instruction description.

        Args:
          letter: A string representing a letter that is expected in the response.
          let_frequency: An integer specifying the number of times `keyword` is
            expected to appear in the response.
          let_relation: A string in (`미만`, `이상`), defining the
            relational operator for comparison. Two relational comparisons are
            supported for now; if '미만', the actual number of
            occurrences < frequency; if '이상', the actual number of
            occurrences >= frequency.

        Returns:
          A string representing the instruction description.
        """
        if (
            not letter
            or len(letter) > 1
            or ord(letter.lower()) < 97
            or ord(letter.lower()) > 122
        ):
            self._letter = random.choice(list(string.ascii_letters))
        else:
            self._letter = letter.strip()
        self._letter = self._letter.lower()

        self._frequency = let_frequency
        if self._frequency is None or self._frequency < 0:
            self._frequency = random.randint(1, _LETTER_FREQUENCY)

        if let_relation is None:
            self._comparison_relation = random.choice(_COMPARISON_RELATION)
        elif let_relation not in _COMPARISON_RELATION:
            raise ValueError(
                "The supported relation for comparison must be in "
                f"{_COMPARISON_RELATION}, but {let_relation} is given."
            )
        else:
            self._comparison_relation = let_relation

        self._description_pattern = (
            "응답에서 '{letter}'자가 {let_frequency}번"
            " {let_relation} 등장해야 합니다."
        )

        return self._description_pattern.format(
            letter=self._letter,
            let_frequency=self._frequency,
            let_relation=self._comparison_relation,
        )

    def get_instruction_args(self):
        """Returns the keyword args of build description."""
        return {
            "letter": self._letter,
            "let_frequency": self._frequency,
            "let_relation": self._comparison_relation,
        }

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["letter", "let_frequency", "let_relation"]

    def check_following(self, value):
        """Checks that the response contains the letter at the right frequency."""
        value = value.lower()
        letters = collections.Counter(value)

        if self._comparison_relation == _COMPARISON_RELATION[0]:
            return letters[self._letter] < self._frequency
        else:
            return letters[self._letter] >= self._frequency


class CapitalLettersEnglishChecker(Instruction):
    """Checks that the response is in english and is in all capital letters."""

    def build_description(self):
        """Build the instruction description."""
        self._description_pattern = (
            "답변은 영문 대문자로만 작성하세요."
        )
        return self._description_pattern

    def get_instruction_args(self):
        return None

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return []

    def check_following(self, value):
        """Checks that the response is in English and in all capital letters."""
        assert isinstance(value, str)

        try:
            return value.isupper() and langdetect.detect(value) == "en"
        except langdetect.LangDetectException as e:
            # Count as instruction is followed.
            logging.error(
                "Unable to detect language for text %s due to %s", value, e
            )  # refex: disable=pytotw.037
            return False


class LowercaseLettersEnglishChecker(Instruction):
    """Checks that the response is in english and is in all lowercase letters."""

    def build_description(self):
        """Build the instruction description."""
        self._description_pattern = (
            "답변을 영문 소문자로만 작성하세요. 대문자는 사용할 수 없습니다."
        )
        return self._description_pattern

    def get_instruction_args(self):
        return None

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return []

    def check_following(self, value):
        """Checks that the response is in English and in all lowercase letters."""
        assert isinstance(value, str)

        try:
            return value.islower() and langdetect.detect(value) == "en"
        except langdetect.LangDetectException as e:
            # Count as instruction is followed.
            logging.error(
                "Unable to detect language for text %s due to %s", value, e
            )  # refex: disable=pytotw.037
            return False


class CommaChecker(Instruction):
    """Checks the response for no commas."""

    def build_description(self):
        """Build the instruction description."""
        self._description_pattern = (
            "답변에 쉼표를 사용하지 마세요."
        )
        return self._description_pattern

    def get_instruction_args(self):
        return None

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return []

    def check_following(self, value):
        """Checks that the response does not contain commas."""
        return not re.search(r"\,", value)


class CapitalWordFrequencyChecker(Instruction):
    """Checks frequency of words with all capital letters."""

    def build_description(
        self,
        capital_frequency=None,
        capital_relation=None,
    ):
        """Build the instruction description.

        Args:
          capital_frequency: An integer that represents the number of words that
            should be in all capital letters.
          capital_relation: A string that is 'at least' or 'at most' that refers to
            the frequency.

        Returns:
          A string representing the instruction description.
        """
        self._frequency = capital_frequency
        if self._frequency is None:
            self._frequency = random.randint(1, _ALL_CAPITAL_WORD_FREQUENCY)

        self._comparison_relation = capital_relation
        if capital_relation is None:
            self._comparison_relation = random.choice(_COMPARISON_RELATION)
        elif capital_relation not in _COMPARISON_RELATION:
            raise ValueError(
                "The supported relation for comparison must be in "
                f"{_COMPARISON_RELATION}, but {capital_relation} is given."
            )

        self._description_pattern = (
            "답변을 영문으로만 작성하고, "
            "모든 철자가 영문 대문자로 된 단어를 {frequency}번 {relation} 사용하세요."
        )

        return self._description_pattern.format(
            frequency=self._frequency, relation=self._comparison_relation
        )

    def get_instruction_args(self):
        """Returns the keyword args of build description."""
        return {
            "capital_frequency": self._frequency,
            "capital_relation": self._comparison_relation,
        }

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["capital_frequency", "capital_relation"]

    def check_following(self, value):
        """Checks the frequency of words with all capital letters."""
        assert isinstance(value, str)

        # 영어 텍스트인지 감지
        try:
            if langdetect.detect(value) != "en":
                return False
        except langdetect.LangDetectException as e:
            logging.error(
                "Unable to detect language for text %s due to %s", value, e
            )
            return False

        # Hyphenated words will count as one word
        words = instructions_util.nltk.word_tokenize(value)
        capital_words = [word for word in words if word.isupper()]

        capital_words = len(capital_words)

        if self._comparison_relation == _COMPARISON_RELATION[0]:
            return capital_words < self._frequency
        else:
            return capital_words >= self._frequency


class QuotationChecker(Instruction):
    """Checks response is wrapped with double quotation marks."""

    def build_description(self):
        """Build the instruction description."""
        self._description_pattern = (
            "답변 전체를 큰따옴표로 감싸세요."
        )
        return self._description_pattern

    def get_instruction_args(self):
        """Returns the keyword args of build description."""
        return None

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return []

    def check_following(self, value):
        """Checks if the response is wrapped with double quotation marks."""
        value = value.strip()
        return len(value) > 1 and value[0] == '"' and value[-1] == '"'


class KoreanLetterFrequencyChecker(Instruction):
    """Check the frequency of Korean initial consonants (초성)."""

    def normalize_consonant(self, value):
        """Normalize consonant characters by mapping them to a unified form."""
        # Mapping of initial consonants to their unified forms
        consonant_mapping = {
            'ᄀ': 'ㄱ', 'ᄂ': 'ㄴ', 'ᄃ': 'ㄷ', 'ᄅ': 'ㄹ',
            'ᄆ': 'ㅁ', 'ᄇ': 'ㅂ', 'ᄉ': 'ㅅ', 'ᄋ': 'ㅇ',
            'ᄌ': 'ㅈ', 'ᄎ': 'ㅊ', 'ᄏ': 'ㅋ', 'ᄐ': 'ㅌ',
            'ᄑ': 'ㅍ', 'ᄒ': 'ㅎ',
        }
        return consonant_mapping.get(value, value)

    def build_description(self, *, letter=None, let_frequency=None, let_relation=None):
        """Build the instruction description.
        
        Args:
          letter: A string representing a Korean initial consonant expect in the response.
          frequency: An integer specifying the number of times 'ko_letter' is 
            expected to appear in the response.
        relation: A string in ('미만', '이상'), defining the realtion operator for comparion.
        Two relational comparisons are supported for now; if '미만', the actual number of
            occurrences < frequency; if '이상', the actual number of occurrences >= frequency.
        Returns:
            A string representing the instruction description.
        """

        consonants = "ㄱㄴㄷㄹㅁㅂㅅㅇㅈㅊㅋㅌㅍㅎ"   
        korean_letters = set(consonants)

        if not letter or len(letter) > 1 or letter not in korean_letters:
            self._letter = random.choice(list(korean_letters))

        else:
            self._letter = letter

        self._frequency = let_frequency
        if self._frequency is None or self._frequency < 0:
            self._frequency = random.randint(1, _LETTER_FREQUENCY)

        if let_relation is None:
            self._comparison_relation = random.choice(_COMPARISON_RELATION)
        elif let_relation not in _COMPARISON_RELATION:
            raise ValueError(
                "The supported relation for comparison must be '미만' or '이상, "
                f"but {let_relation} is given."
            )
        else:
            self._comparison_relation = let_relation

        self._description_pattern = (
            "응답에서 '{letter}'자가 초성에 {frequency} 번"
            " {relation} 등장해야 합니다."
        )
        return self._description_pattern.format(
            letter = self._letter,
            frequency=self._frequency,
            relation=self._comparison_relation,
        )

    def get_instruction_args(self):
        """Returns the keyword args of build description."""
        return {
            "letter": self._letter,
            "let_frequency": self._frequency,
            "let_relation": self._comparison_relation,
        }

    def get_instruction_args_keys(self):
        """Returns the args keys of 'build_description."""
        return ["letter", "let_frequency", "let_relation"]

    def check_following(self, value):
        """Checks that the response contains the letter at the right frequency."""

        decomposed_text = unicodedata.normalize("NFKD", value)
        normalized_text = "".join(self.normalize_consonant(char) for char in decomposed_text)

        letters = collections.Counter(normalized_text)

        if self._comparison_relation == _COMPARISON_RELATION[0]:
            return letters[self._letter] < self._frequency
        else:
            return letters[self._letter] >= self._frequency


class MultipleChoiceChecker(Instruction):
    """Checks if the response matches exactly one of the multiple-choice options."""

    def build_description(self, *, options=None):
        """Build the instruction description."""
        if options is None:
            self._options =_MULTIPLE_OPTIONS
        else:
            self._options = list(set(options))
        self._description_pattern = (
            "답변은 {options} 중 하나로만 작성하세요. 다른 내용은 포함하지 마세요."
        )
        return self._description_pattern.format(options=self._options)

    def get_instruction_args(self):
        """Returns the keyword args of `build_description`."""
        return {"options": self._options}

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["options"]

    def check_following(self, value):
        """Checks if the response matches exactly one of the options.
        Args:
          value: A string representing the response.
        Returns:
          True if the response matches exactly one option from the choices; otherwise, False.
        """
        if value in self._options:
            return True
        return False


class ChangeSpecificKeywordOnly(Instruction):
    """Check if only the instructed keyword has been changed and nothing else has been modified."""
    
    def build_description(self, *, prompt_to_change=None, keyword_from=None, keyword_to=None):
        """Build the instruction description.
        
        Args:
          prompt_to_change: The prompt to be partially modified.
          keyword_from: A string representing a keyword in `prompt_to_change`
            that is to be replaced.
          keyword_to: Another string representing a keyword that replaces
            `keyword_from` in `prompt_to_change`.
        
        Returns:
          A string representing the instruction description.
        """
        if not prompt_to_change:
            raise ValueError("prompt_to_change must be set.")
        else:
            self._prompt_to_change = prompt_to_change
        
        if not keyword_from:
            raise ValueError("keyword_from must be set.")
        elif keyword_from not in prompt_to_change:
            raise ValueError("keyword_from is not in the prompt.")
        else:
            self._keyword_from = keyword_from
        
        if not keyword_to:
            raise ValueError("keyword_to must be set.")
        else:
            self._keyword_to = keyword_to

        def final_consonant(word):
            """한글 글자의 받침 번호를 반환합니다. 받침이 없으면 0, ㄹ이면 8을 반환합니다."""
            char = word[-1]
            # 한글 범위 내에 있는지 확인
            if '가' <= char <= '힣':
                code = ord(char) - ord('가')
                # 종성 번호
                jongseong = code % 28
                return jongseong
            # 한글이 아니면 : 조사 아무거나 ('를', '로')
            else:
                return 0
            
        def proper_particle_obj(keyword_from):
            jongseong = final_consonant(keyword_from)
            if jongseong == 0:
                particle_obj = '를'
            else:
                particle_obj = '을'
            return particle_obj
        
        def proper_particle_to(keyword_to):
            jongseong = final_consonant(keyword_to)
            if jongseong == 0:
                particle_to = '로'
            elif jongseong == 8:
                particle_to = '로'
            else:
                particle_to = '으로'
            return particle_to

        self._description_pattern = (
            "위 글에서 단어 '{keyword_from}'{particle_obj} 찾아 "
            "모두 '{keyword_to}'{particle_to} 바꾸세요. "
            "다른 부분은 변경하지 말고 그대로 유지하세요."
        )
        return self._description_pattern.format(
            keyword_from=self._keyword_from, particle_obj=proper_particle_obj(self._keyword_from),
            keyword_to=self._keyword_to, particle_to=proper_particle_to(self._keyword_to)
        )

    def get_instruction_args(self):
        """Returns the keyword args of `build_description`."""
        return {
            "prompt_to_change" : self._prompt_to_change,
            "keyword_from" : self._keyword_from,
            "keyword_to" : self._keyword_to
        }

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["prompt_to_change", "keyword_from", "keyword_to"]

    def check_following(self, value):
        """Check if only the instructed keyword has been changed and nothing else has been modified.
        
        Args:
          value: A string representing the response.

        Returns:
          True if replacing `keyword_from` with `keyword_to` in `prompt_to_change` results in `value`;
          otherwise False.
        """
        changed_prompt = self._prompt_to_change.replace(self._keyword_from, self._keyword_to)
        return changed_prompt == value


class NumberLettersIncluded(Instruction):
    """Checks the number of letters in Korean, space included."""

    def build_description(self, *, num_letters=None, relation=None):
        """Build the instruction description.

        Args:
          num_letters: An integer specifying the number of letters contained in the
            response.
          relation: A string in (`미만`, `이상`), defining the relational
            operator for comparison.
            Two relational comparisons are supported for now:
            if '미만', the actual number of letters < num_letters;
            if '이상', the actual number of letters >= num_letters.

        Returns:
          A string representing the instruction description.
        """

        self._num_letters = num_letters
        if self._num_letters is None or self._num_letters < 0:
            self._num_letters = random.randint(
                _NUM_LETTERS_LOWER_LIMIT, _NUM_LETTERS_UPPER_LIMIT
            )

        if relation is None:
            self._comparison_relation = random.choice(_COMPARISON_RELATION)
        elif relation not in _COMPARISON_RELATION:
            raise ValueError(
                "The supported relation for comparison must be in "
                f"{_COMPARISON_RELATION}, but {relation} is given."
            )
        else:
            self._comparison_relation = relation

        self._description_pattern = "공백 포함 {num_letters}자 {relation}의 답변을 작성하며, 한국어로만 답변하세요."

        return self._description_pattern.format(
            relation=self._comparison_relation, num_letters=self._num_letters
        )

    def get_instruction_args(self):
        """Returns the keyword args of `build_description`."""
        return {"num_letters": self._num_letters, "relation": self._comparison_relation}

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["num_letters", "relation"]

    def check_following(self, value):
        """Checks if the response contains the expected number of letters."""
        num_letters = instructions_util.count_letters_included(value)

        if self._comparison_relation == _COMPARISON_RELATION[0]:
            return num_letters < self._num_letters
        elif self._comparison_relation == _COMPARISON_RELATION[1]:
            return num_letters >= self._num_letters


class NumberLettersExcluded(Instruction):
    """Checks the number of letters in Korean, space excluded."""

    def build_description(self, *, num_letters=None, relation=None):
        """Build the instruction description.

        Args:
          num_letters: An integer specifying the number of letters contained in the
            response.
          relation: A string in (`미만`, `이상`), defining the relational
            operator for comparison.
            Two relational comparisons are supported for now:
            if '미만', the actual number of letters < num_letters;
            if '이상', the actual number of letters >= num_letters.

        Returns:
          A string representing the instruction description.
        """

        self._num_letters = num_letters
        if self._num_letters is None or self._num_letters < 0:
            self._num_letters = random.randint(
                _NUM_LETTERS_LOWER_LIMIT, _NUM_LETTERS_UPPER_LIMIT
            )

        if relation is None:
            self._comparison_relation = random.choice(_COMPARISON_RELATION)
        elif relation not in _COMPARISON_RELATION:
            raise ValueError(
                "The supported relation for comparison must be in "
                f"{_COMPARISON_RELATION}, but {relation} is given."
            )
        else:
            self._comparison_relation = relation

        self._description_pattern = "공백 제외 {num_letters}자 {relation}의 답변을 작성하며, 한국어로만 답변하세요."

        return self._description_pattern.format(
            relation=self._comparison_relation, num_letters=self._num_letters
        )

    def get_instruction_args(self):
        """Returns the keyword args of `build_description`."""
        return {"num_letters": self._num_letters, "relation": self._comparison_relation}

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["num_letters", "relation"]

    def check_following(self, value):
        """Checks if the response contains the expected number of letters."""
        num_letters = instructions_util.count_letters_excluded(value)

        if self._comparison_relation == _COMPARISON_RELATION[0]:
            return num_letters < self._num_letters
        elif self._comparison_relation == _COMPARISON_RELATION[1]:
            return num_letters >= self._num_letters
