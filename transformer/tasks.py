# Copyright 2022 Google.
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

"""Add Tasks to registry."""

import functools

from transformer import text_dataset
import seqio
import tensorflow as tf


TaskRegistry = seqio.TaskRegistry


def define_pg19_task(name: str, vocab: seqio.Vocabulary):
  seqio.TaskRegistry.add(
      name,
      seqio.TfdsDataSource(
          tfds_name="pg19:0.1.1"
      ),
      preprocessors=[
          functools.partial(text_dataset.rekey_articles,
                            rekey={"book_text": "targets"},
                            keep={"book_title", "book_id", "publication_date"}),
          seqio.preprocessors.tokenize,
      ],
      output_features={
          "targets": seqio.Feature(vocab,
                                   add_eos=False, dtype=tf.int32),
      }
  )


def define_long_pile_task(name: str, vocab: seqio.Vocabulary):
  seqio.TaskRegistry.add(
      name,
      seqio.TfdsDataSource(tfds_name='long_pile:1.1.0'),
      preprocessors=[
          functools.partial(text_dataset.rekey_articles,
                            rekey={'text': 'targets'},
                            keep={'subset'}), seqio.preprocessors.tokenize,
          seqio.preprocessors.append_eos
      ],
      output_features={
          'targets': seqio.Feature(vocab, add_eos=True, dtype=tf.int32),
      })


define_pg19_task("pg19_bytes", seqio.ByteVocabulary())
define_pg19_task("pg19_tokens", seqio.SentencePieceVocabulary('vocabs/pg19train_bpe_32000.model'))
define_long_pile_task('long_pile_tokens', seqio.SentencePieceVocabulary('tokenizers/long_pile_48k.model'))
