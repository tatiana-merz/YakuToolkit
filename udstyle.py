import os
import sys
import wget
import subprocess
from math import log, sqrt
from contextlib import contextmanager
from collections import Counter
import pandas as pd

from openpyxl import load_workbook

ID, FORM, LEMMA, UPOS, XPOS, FEATS, HEAD, DEPREL, DEPS, MISC = range(10)

def which(program, exception=True):
	"""Return first match for program in search path.

	:param exception: By default, ValueError is raised when program not found.
		Pass False to return None in this case."""
	for path in os.environ.get('PATH', os.defpath).split(":"):
		if path and os.path.exists(os.path.join(path, program)):
			return os.path.join(path, program)
	if exception:
		raise ValueError('%r not found in path; please install it.' % program)


@contextmanager
def genericdecompressor(cmd, filename, encoding='utf8'):
	"""Run command line decompressor on file and return file object.

	:param cmd: executable in path with gzip-like command line interface;
		e.g., ``gzip, zstd, lz4, bzip2, lzop``
	:param filename: the file to decompress.
	:param encoding: if None, mode is binary; otherwise, text.
	:raises ValueError: if command returns an error.
	:returns: a file-like object that must be used in a with-statement;
		supports .read() and iteration, but not seeking."""
	with subprocess.Popen(
			[which(cmd), '--decompress', '--stdout', '--quiet', filename],
			stdout=subprocess.PIPE, stderr=subprocess.PIPE,
			encoding=encoding) as proc:
		# FIXME: should use select to avoid deadlocks due to OS pipe buffers
		# filling up and blocking the child process.
		yield proc.stdout
		retcode = proc.wait()
		if retcode:  # FIXME: retcode 2 means warning. allow warnings?
			raise ValueError('non-zero exit code %s from compressor %s:\n%r'
					% (retcode, cmd, proc.stderr.read()))


def openread(filename, encoding='utf8'):
	"""Open stdin/file for reading; decompress gz/lz4/zst files on-the-fly.

	:param encoding: if None, mode is binary; otherwise, text."""
	mode = 'rb' if encoding is None else 'rt'
	if filename == '-':  # TODO: decompress stdin on-the-fly
		return open(sys.stdin.fileno(), mode=mode, encoding=encoding)
	if not isinstance(filename, int):
		if filename.endswith('.gz'):
			return genericdecompressor('gzip', filename, encoding)
		elif filename.endswith('.zst'):
			return genericdecompressor('zstd', filename, encoding)
		elif filename.endswith('.lz4'):
			return genericdecompressor('lz4', filename, encoding)
	return open(filename, mode=mode, encoding=encoding)


def parsefiles(filenames, lang):
	"""Parse UTF-8 encoded plain text files with Stanza if a corresponding
	.conllu file does not exist already."""
	nlp = None
	newfilenames = []
	for filename in filenames:
		conllu = '%s.conllu' % os.path.splitext(filename)[0]
		newfilenames.append(conllu)
		if (not os.path.exists(conllu)
				or os.stat(conllu).st_mtime < os.stat(filename).st_mtime):
			if nlp is None:
				import stanza
				from stanza.utils.conll import CoNLL
				processors = 'tokenize,mwt,pos,lemma,depparse'
				try:
					nlp = stanza.Pipeline(lang, processors=processors)
				except FileNotFoundError:
					stanza.download(lang)
					nlp = stanza.Pipeline(lang, processors=processors)
			with open(filename, encoding='utf8') as inp:
				doc = nlp(inp.read())
			# TODO: preserve paragraph breaks
			CoNLL.write_doc2conll(doc, conllu)
	return newfilenames


def conllureader(filename, excludepunct=False):
	"""Load corpus. Returns list of lists of lists:
	sentences[sentno][tokenno][fieldno]"""
	result = []
	sent = []
	with openread(filename) as inp:
		for line in inp:
			if line == '\n':
				if sent:
					try:
						result.append(renumber(sent))
					except KeyError:
						pass
					sent = []
			elif line.startswith('#'):  # ignore all comments
				pass
			else:
				fields = line[:-1].split('\t')
				if '.' in fields[ID]:  # skip empty nodes
					continue
				elif excludepunct and fields[UPOS] == 'PUNCT':
					continue
				elif '-' in fields[ID]:  # multiword tokens
					fields[ID] = int(fields[ID][:fields[ID].index('-')])
				else:  # normal tokens
					fields[ID] = int(fields[ID])
				try:
					fields[HEAD] = int(fields[HEAD])
				except ValueError:
					continue
				sent.append(fields)
	if not result:
		raise ValueError('no sentences; not a valid .conllu file?')
	return result


def renumber(sent):
	"""Fix non-contiguous IDs because of multiword tokens or removed tokens"""
	mapping = {line[ID]: n for n, line in enumerate(sent, 1)}
	mapping[0] = 0
	for line in sent:
		line[ID] = mapping[line[ID]]
		line[HEAD] = mapping[line[HEAD]]
	return sent


def mean(iterable):
	"""Arithmetic mean."""
	seq = list(iterable)  # accept generators
	return sum(seq) / len(seq)


def analyze(filename, excludepunct=True):
	"""Return a dict {featname: vector, ...} describing UD file.
	Each feature vector has a value for each sentence."""
	sentences = conllureader(filename, excludepunct=excludepunct)
	result = {}
	result['LEN'] = [len(sent) for sent in sentences]
	# Ignore certain relations, following Chen and Gerdes (2017, p. 57)
	# http://www.aclweb.org/anthology/W17-6508
	exclude = ('fixed', 'flat', 'conj', 'punct')
	# Gibson (1998) http://dx.doi.org/10.1016/S0010-0277(98)00034-1
	# Liu (2008) https://hdl.handle.net/10371/70907
	# mean dependency distance
	result['MDD'] = [
			mean(abs(line[ID] - line[HEAD]) for line in sent
				if line[DEPREL] not in exclude)
			for sent in sentences]
	# Lei & Jockers (2018): https://doi.org/10.1080/09296174.2018.1504615
	# normalized dependency distance
	result['NDD'] = [
		abs(log(mdd
				/ sqrt(
					([line[DEPREL] for line in sent].index('root') + 1)
					* len(sent))))
		for mdd, sent in zip(result['MDD'], sentences)]
	# proportion of adjacent dependencies
	# https://doi.org/10.1016/j.langsci.2016.09.006
	result['ADJ'] = [mean(abs(line[ID] - line[HEAD]) == 1 for line in sent)
			for sent in sentences]
	# dependency direction: proportion of left dependents
	# http://www.aclweb.org/anthology/W17-6508
	result['LEFT'] = [mean(line[ID] < line[HEAD] for line in sent)
			for sent in sentences]
	# nominal modifiers;
	# attempt to measure phrasal complexity (as opposed to clausal complexity).
	# see e.g. https://doi.org/10.1016/j.jeap.2010.01.001
	result['MOD'] = [mean(line[DEPREL] == 'nmod' for line in sent)
			for sent in sentences]
	# number of clauses per sentence; https://doi.org/10.1007/s11145-007-9107-5
	result['CLS'] = [1 + sum(line[UPOS] == 'VERB' for line in sent)
			for sent in sentences]
	# avg clause len (clauses/words) https://aclanthology.org/2020.lrec-1.883
	result['CLL'] = [
			len(sent) / max(1, sum(line[UPOS] == 'VERB' for line in sent))
			for sent in sentences]
	# lexical density: ratio of content words over total number of words
	# https://aclanthology.org/2020.lrec-1.883
	content = ('ADJ', 'ADV', 'INTJ', 'NOUN', 'PROPN', 'VERB')
	result['LXD'] = [sum(line[UPOS] in content for line in sent) / len(sent)
			for sent in sentences]
	# This gives a macro average over the per-sentence scores.
	# Might want to look at standard deviation and other aspects of the
	# distribution. TODO: offer micro average as well.
	for a, b in result.items():
		result[a] = sum(b) / len(b)
	result.update(counttags(sentences))
	return result


def counttags(sentences):
	"""Count POS and dependency tags; returns relative frequencies."""
	numtokens = sum(len(sent) for sent in sentences)
	postags = Counter(line[UPOS] for sent in sentences for line in sent)
	deptags = Counter(line[DEPREL] for sent in sentences for line in sent)
	tags = {a: postags[a] / numtokens for a in [
			'ADJ',  # adjective
			'ADP',  # adposition
			'ADV',  # adverb
			'AUX',  # auxiliary
			'CCONJ',  # coordinating conjunction
			'DET',  # determiner
			'INTJ',  # interjection
			'NOUN',  # noun
			'NUM',  # numeral
			'PART',  # particle
			'PRON',  # pronoun
			'PROPN',  # proper noun
			'PUNCT',  # punctuation
			'SCONJ',  # subordinating conjunction
			'SYM',  # symbol
			'VERB',  # verb
			'X',  # other
			]}
	tags.update({a: deptags[a] / numtokens for a in [
			'acl',  # clausal modifier of noun (adnominal clause)
			'acl:relcl',  # relative clause modifier
			'advcl',  # adverbial clause modifier
			'advmod',  # adverbial modifier
			'advmod:emph',  # emphasizing word, intensifier
			'advmod:lmod',  # locative adverbial modifier
			'amod',  # adjectival modifier
			'appos',  # appositional modifier
			'aux',  # auxiliary
			'aux:pass',  # passive auxiliary
			'case',  # case marking
			'cc',  # coordinating conjunction
			'cc:preconj',  # preconjunct
			'ccomp',  # clausal complement
			'clf',  # classifier
			'compound',  # compound
			'compound:lvc',  # light verb construction
			'compound:prt',  # phrasal verb particle
			'compound:redup',  # reduplicated compounds
			'compound:svc',  # serial verb compounds
			'conj',  # conjunct
			'cop',  # copula
			'csubj',  # clausal subject
			'csubj:pass',  # clausal passive subject
			'dep',  # unspecified dependency
			'det',  # determiner
			'det:numgov',  # pronominal quantifier governing the case of the noun
			'det:nummod',  # pronominal quantifier agreeing in case with the noun
			'det:poss',  # possessive determiner
			'discourse',  # discourse element
			'dislocated',  # dislocated elements
			'expl',  # expletive
			'expl:impers',  # impersonal expletive
			'expl:pass',  # reflexive pronoun used in reflexive passive
			'expl:pv',  # reflexive clitic with an inherently reflexive verb
			'fixed',  # fixed multiword expression
			'flat',  # flat multiword expression
			'flat:foreign',  # foreign words
			'flat:name',  # names
			'goeswith',  # goes with
			'iobj',  # indirect object
			'list',  # list
			'mark',  # marker
			'nmod',  # nominal modifier
			'nmod:poss',  # possessive nominal modifier
			'nmod:tmod',  # temporal modifier
			'nsubj',  # nominal subject
			'nsubj:pass',  # passive nominal subject
			'nummod',  # numeric modifier
			'nummod:gov',  # numeric modifier governing the case of the noun
			'obj',  # object
			'obl',  # oblique nominal
			'obl:agent',  # agent modifier
			'obl:arg',  # oblique argument
			'obl:lmod',  # locative modifier
			'obl:tmod',  # temporal modifier
			'orphan',  # orphan
			'parataxis',  # parataxis
			'punct',  # punctuation
			'reparandum',  # overridden disfluency
			'root',  # root
			'vocative',  # vocative
			'xcomp',  # open clausal complement
			]})
	return tags



def conllu_download():
	path = './Turkic_UDs'

	urls = ['https://raw.githubusercontent.com/UniversalDependencies/UD_Yakut-YKTDT/master/sah_yktdt-ud-test.conllu',
			'https://raw.githubusercontent.com/UniversalDependencies/UD_Tatar-NMCTT/master/tt_nmctt-ud-test.conllu',
			'https://raw.githubusercontent.com/UniversalDependencies/UD_Kazakh-KTB/master/kk_ktb-ud-train.conllu',
		'https://raw.githubusercontent.com/UniversalDependencies/UD_Turkish-Kenet/master/tr_kenet-ud-dev.conllu',
		'https://raw.githubusercontent.com/UniversalDependencies/UD_Uyghur-UDT/master/ug_udt-ud-dev.conllu',
		'https://raw.githubusercontent.com/UniversalDependencies/UD_Old_Turkish-Tonqq/master/otk_tonqq-ud-test.conllu',
		'https://raw.githubusercontent.com/UniversalDependencies/UD_Tupinamba-TuDeT/dev/tpn_tudet-ud-test.conllu',
		'https://raw.githubusercontent.com/UniversalDependencies/UD_Guarani-OldTuDeT/dev/gn_oldtudet-ud-test.conllu']

	for url in urls:
		filename = path + '/' + os.path.basename(url) # get the full path of the file
		if os.path.exists(filename):
			os.remove(filename) # if exist, remove it directly
		wget.download(url, out=filename) # download it to the specific path.    

def result_print_all():
	
	yakut = analyze("./Turkic_UDs/sah_yktdt-ud-test.conllu")
	tatar = analyze("./Turkic_UDs/tt_nmctt-ud-test.conllu")
	kazakh = analyze("./Turkic_UDs/kk_ktb-ud-train.conllu")
	turkish = analyze("./Turkic_UDs/tr_kenet-ud-dev.conllu")
	old_tu = analyze("./Turkic_UDs/otk_tonqq-ud-test.conllu")
	uygur = analyze("./Turkic_UDs/ug_udt-ud-dev.conllu")	
	df = pd.DataFrame.from_dict([yakut, tatar, kazakh, turkish, old_tu, uygur])
	pd.options.display.float_format = '{:.3f}'.format
	
	df.insert(loc=0, column="Metrics", value=["Yakut", "Tatar", "Kazakh", "Turkish", "Old_Turkish", "Uygur"])
	df.to_csv('./Turkic_UDs/UD_Tr_metrics.csv', index=False, header=True, float_format='%.3f')
	return df

if __name__ == '__main__':
	conllu_download()
	print(result_print_all())
