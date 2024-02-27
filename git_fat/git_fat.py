#!/usr/bin/env python
# -*- mode:python -*-

from __future__ import print_function, with_statement

import argparse
import configparser as cfgparser
import errno
import hashlib
import io
import logging as _logging  # Use logger.error(), not logging.error()
import os
import platform
import shutil
import stat
import subprocess as sub
import sys
import tempfile
import warnings
from urllib.error import URLError
from urllib.request import (
    HTTPBasicAuthHandler, HTTPPasswordMgrWithDefaultRealm, build_opener, urlopen)
from typing import (
    TYPE_CHECKING, Any, Callable, Container, Dict, FrozenSet, IO, Iterable, Iterator, List,
    Optional, Tuple, cast)

_logging.basicConfig(format='%(levelname)s:%(filename)s: %(message)s')
logger = _logging.getLogger(__name__)


__version__ = '0.3.4'

BLOCK_SIZE = 4096

NOT_IMPLEMENTED_MESSAGE = "This method isn't implemented for this backend!"


def get_log_level(log_level_string: str) -> int:
    log_level_string = log_level_string.lower()

    if not log_level_string:
        return _logging.WARNING

    levels = {'debug': _logging.DEBUG,
              'info': _logging.INFO,
              'warning': _logging.WARNING,
              'error': _logging.ERROR,
              'critical': _logging.CRITICAL}

    if log_level_string in levels:
        return levels[log_level_string]

    logger.warning(f'Invalid log level: {log_level_string}')
    return _logging.WARNING


GIT_FAT_LOG_LEVEL = get_log_level(os.getenv("GIT_FAT_LOG_LEVEL", ""))
GIT_FAT_LOG_FILE = os.getenv("GIT_FAT_LOG_FILE", "")
GIT_SSH = os.getenv("GIT_SSH")


def git(cliargs: List[str], *args: Any, **kwargs: Any) -> sub.Popen:
    ''' Calls git commands with Popen arguments '''
    binary = kwargs.pop('binary', False)

    if not binary:
        kwargs.update({'encoding': 'utf-8'})

    if GIT_FAT_LOG_FILE and "--failfast" in sys.argv:
        # Flush any prior logger warning/error/critical to the log file
        # which is being checked by unit tests.
        sys.stdout.flush()
        sys.stderr.flush()
    if GIT_FAT_LOG_LEVEL == _logging.DEBUG:
        cmd = ' '.join(['git'] + cliargs)
        logger.debug(f'{cmd} ({args}, {kwargs})')

    return sub.Popen(['git'] + cliargs, *args, **kwargs)


if not TYPE_CHECKING:
    def check_output2(args, **kwargs):
        if GIT_FAT_LOG_FILE and "--failfast" in sys.argv:
            # Flush any prior logger warning/error/critical to the log file
            # which is being checked by unit tests.
            sys.stdout.flush()
            sys.stderr.flush()
        if GIT_FAT_LOG_LEVEL == _logging.DEBUG:
            args2 = args
            for i, v in enumerate(args):
                args[i] = v.replace("\x00", r"\x00")
            logger.debug(' '.join(args2))
        return original_check_output(args, **kwargs)

    original_check_output = sub.check_output
    sub.check_output = check_output2


def mkdir_p(path: str) -> None:
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


# -----------------------------------------------------------------------------
# On Windows files may be read only and may require changing
# permissions. Always use these functions for moving/deleting files.

def move_file(src: str, dst: str) -> None:
    if platform.system() == "Windows":
        if os.path.exists(src) and not os.access(src, os.W_OK):
            st = os.stat(src)
            os.chmod(src, st.st_mode | stat.S_IWUSR)
        if os.path.exists(dst) and not os.access(dst, os.W_OK):
            st = os.stat(dst)
            os.chmod(dst, st.st_mode | stat.S_IWUSR)
    shutil.move(src, dst)


def delete_file(f: str) -> None:
    if platform.system() == "Windows":
        if os.path.exists(f) and not os.access(f, os.W_OK):
            st = os.stat(f)
            os.chmod(f, st.st_mode | stat.S_IWUSR)
    os.remove(f)

# -----------------------------------------------------------------------------


def make_sys_streams_binary() -> None:
    # Information for future: in Python 3 use sys.stdin.detach()
    # for both Linux and Windows.
    if platform.system() == "Windows":
        import msvcrt  # pylint: disable=import-error,import-outside-toplevel
        result = msvcrt.setmode(sys.stdin.fileno(), os.O_BINARY)  # type: ignore[attr-defined]
        if result == -1:
            raise Exception("Setting sys.stdin to binary mode failed")
        result = msvcrt.setmode(sys.stdout.fileno(), os.O_BINARY)  # type: ignore[attr-defined]
        if result == -1:
            raise Exception("Setting sys.stdout to binary mode failed")


def umask() -> int:
    '''
    Get umask without changing it.
    '''
    old = os.umask(0)
    os.umask(old)
    return old


def readblocks(stream: IO[bytes]) -> Iterable[bytes]:
    '''
    Reads BLOCK_SIZE from stream and yields it
    '''
    while True:
        data = stream.read(BLOCK_SIZE)
        if not data:
            break
        yield data


def cat_iter(initer: Iterable[bytes], outstream: IO[bytes]) -> None:
    for block in initer:
        outstream.write(block)


def cat(instream: IO[bytes], outstream: IO[bytes]) -> None:
    return cat_iter(readblocks(instream), outstream)


def gitconfig_get(name: str, cfgfile: Optional[str] = None) -> str:
    args = ['config', '--get']
    if cfgfile is not None:
        args += ['--file', cfgfile]
    args.append(name)
    p = git(args, stdout=sub.PIPE)
    output = p.communicate()[0].strip()
    if p.returncode != 0:
        return ''

    return output


def gitconfig_set(name, value, cfgfile=None):
    args = ['git', 'config']
    if cfgfile is not None:
        args += ['--file', cfgfile]
    args += [name, value]
    sub.check_call(args)


def _config_path(path: Optional[str] = None) -> str:
    try:
        root = sub.check_output('git rev-parse --show-toplevel'.split(),
                                encoding='utf-8').strip()
    except sub.CalledProcessError as exc:
        raise RuntimeError('git-fat must be run from a git directory') from exc
    default_path = os.path.join(root, '.gitfat')
    path = path or default_path
    return path


def _obj_dir() -> str:
    try:
        gitdir = sub.check_output('git rev-parse --git-dir'.split(),
                                  encoding='utf-8').strip()
    except sub.CalledProcessError as exc:
        raise RuntimeError('git-fat must be run from a git directory') from exc
    objdir = os.path.join(gitdir, 'fat', 'objects')
    return objdir


def http_get(baseurl, filename, user=None, password=None):
    ''' Returns file descriptor for http file stream, catches urllib errors '''
    try:
        print(f'Downloading: {filename}')
        geturl = '/'.join([baseurl, filename])
        if user is None:
            res = urlopen(geturl)  # pylint: disable=consider-using-with
        else:
            mgr = HTTPPasswordMgrWithDefaultRealm()
            mgr.add_password(None, baseurl, user, password)
            handler = HTTPBasicAuthHandler(mgr)
            opener = build_opener(handler)
            res = opener.open(geturl)
        return res.fp
    except URLError as e:
        logger.warning(f'{e.reason}: {geturl}')
        return None


def hash_stream(blockiter, outstream):
    '''
    Writes blockiter to outstream and returns the digest and bytes written
    '''
    hasher = hashlib.new('sha1')
    bytes_written = 0

    for block in blockiter:
        # Add the block to be hashed
        if isinstance(block, str):
            block = block.encode('utf-8')

        hasher.update(block)
        bytes_written += len(block)
        outstream.write(block)
    outstream.flush()
    return hasher.hexdigest(), bytes_written


class BackendInterface:
    """ __init__ and pull_files are required, push_files is optional """

    def __init__(self, base_dir, **kwargs):
        """ Configuration options should be set in here """
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)

    def push_files(self, file_list):
        """ Return True if push was successful, False otherwise. Not required but useful """
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)

    def pull_files(self, file_list):
        """ Return True if pull was successful, False otherwise """
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)


class CopyBackend(BackendInterface):
    def __init__(self, base_dir, **kwargs):
        other_path = kwargs.get('remote')
        if not os.path.isdir(other_path):
            raise RuntimeError(
                f'copybackend target path is not directory: {other_path}')
        logger.debug(f'CopyBackend: other_path={other_path}, base_dir={base_dir}')
        self.other_path = other_path
        self.base_dir = base_dir

    def pull_files(self, file_list):
        for f in file_list:
            fullpath = os.path.join(self.other_path, f)
            shutil.copy2(fullpath, self.base_dir)
        return True

    def push_files(self, file_list):
        for f in file_list:
            fullpath = os.path.join(self.base_dir, f)
            shutil.copy2(fullpath, self.other_path)
        return True


class HTTPBackend(BackendInterface):
    """ Pull files from an HTTP server """

    def __init__(self, base_dir, **kwargs):
        remote_url = kwargs.get('remote')
        if not remote_url:
            raise RuntimeError('No remote url configured for http backend')

        if not (remote_url.startswith('http') or remote_url.startswith('https')):
            raise RuntimeError('http remote url must start with http:// or https://')

        self.remote_url = remote_url
        self.user = kwargs.get('user')
        self.password = kwargs.get('password')
        self.base_dir = base_dir

    def pull_files(self, file_list):
        is_success = True

        for o in file_list:
            stream = http_get(self.remote_url, o, self.user, self.password)
            blockiter = readblocks(stream)

            # HTTP Error
            if blockiter is None:
                is_success = False
                continue

            fd, tmpname = tempfile.mkstemp(dir=self.base_dir)
            with os.fdopen(fd, 'wb') as tmpstream:
                # Hash the input, write to temp file
                digest, _ = hash_stream(blockiter, tmpstream)

            if digest != o:
                # Should I retry?
                logger.error(
                    f'Downloaded digest ({digest}) did not match stored digest for orphan: {o}')
                delete_file(tmpname)
                is_success = False
                continue

            objfile = os.path.join(self.base_dir, digest)
            os.chmod(tmpname, int('444', 8) & ~umask())
            # Rename temp file.
            move_file(tmpname, objfile)

        return is_success


class RSyncBackend(BackendInterface):
    """ Push and pull files from rsync remote """

    def __init__(self, base_dir, **kwargs):
        remote_url = kwargs.get('remote')

        # Allow support for rsyncd servers (Looks like "remote = example.org::mybins")
        ssh_user = ''
        ssh_port = ''
        if "::" in remote_url:
            self.is_rsyncd_remote = True
        else:
            self.is_rsyncd_remote = False
            ssh_user = kwargs.get('sshuser')
            ssh_port = kwargs.get('sshport', '22')

        if not remote_url:
            raise RuntimeError("No remote url configured for rsync")

        self.remote_url = remote_url
        self.ssh_user = ssh_user
        self.ssh_port = ssh_port
        self.base_dir = base_dir
        # Swap Windows style drive letters (e.g. 't:') for cygwin style drive letters (e.g. '/t')
        # Otherwise, when using an rsyncd remote (e.g. 'example.org::bin'),
        # The rsync client on Windows will exit with this error:
        # "The source and destination cannot both be remote."
        # Presumably, this is because rsync assumes any path is remote if it contains a colon.
        if platform.system() == 'Windows' and self.is_rsyncd_remote and self.base_dir.find(':') == 1:
            self.base_dir = "/" + self.base_dir[0] + self.base_dir[2:]

    def _rsync(self, push):
        ''' Construct the rsync command '''
        if platform.system() == 'Windows':
            # Windows installer ships its own rsync tool
            rsync_tool = 'git-fat_rsync.exe'
        else:
            rsync_tool = 'rsync'
        cmd_tmpl = [rsync_tool] + ' -s --progress'\
            ' --ignore-existing --from0 --files-from=-'.split()

        if push:
            src, dst = self.base_dir, self.remote_url
        else:
            src, dst = self.remote_url, self.base_dir
        cmd = cmd_tmpl + [src + '/', dst + '/']

        # extra must be passed in as single argv, which is why it's
        # not in the template and split isn't called on it
        if self.is_rsyncd_remote:
            extra = ''
        elif GIT_SSH:
            extra = f'--rsh={GIT_SSH}'
        elif platform.system() == "Windows":
            extra = '--rsh=git-fat_ssh.exe'
        else:
            extra = '--rsh=ssh'

        if self.ssh_user:
            extra = ' '.join([extra, f'-l {self.ssh_user}'])
        if self.ssh_port:
            extra = ' '.join([extra, f'-p {self.ssh_port}'])

        if extra:
            cmd.append(extra)

        return cmd

    def pull_files(self, file_list):
        rsync = self._rsync(push=False)
        logger.debug(f'rsync pull command: {" ".join(rsync)}')
        try:
            with sub.Popen(rsync, stdin=sub.PIPE) as p:
                p.communicate(input=b'\x00'.join(f.encode('utf-8') for f in file_list))
        except OSError as exc:
            # re-raise with a more useful message
            cmd = ' '.join(rsync)
            raise OSError(f'Error running "{cmd}"') from exc

        # TODO: fix for success check
        return True

    def push_files(self, file_list):
        rsync = self._rsync(push=True)
        logger.debug(f'rsync push command: {" ".join(rsync)}')
        with sub.Popen(rsync, stdin=sub.PIPE) as p:
            p.communicate(input=b'\x00'.join(f.encode('utf-8') for f in file_list))
        # TODO: fix for success check
        return True


BACKEND_MAP = {
    'rsync': RSyncBackend,
    'http': HTTPBackend,
    'copy': CopyBackend,
}


class GitFat:

    def __init__(self, backend, full_history=False):

        # The backend instance we use to get the files
        self.backend = backend
        self.full_history = full_history
        self.rev = None  # Unused
        self.objdir = _obj_dir()
        self._cookie = '#$# git-fat '
        self._format = self._cookie + '{digest} {size:20d}\n'

        # Legacy format support below, need to actually check the version once/if we have more than 2
        if os.environ.get('GIT_FAT_VERSION'):
            self._format = self._cookie + '{digest}\n'

        # considers the git-fat version when generating the magic length
        def magiclen(fn: Callable[[str, int], str]) -> int:
            return len(fn(hashlib.sha1(b'dummy').hexdigest(), 5))

        self._magiclen = magiclen(self._encode)

        self.configure()

    def configure(self) -> None:
        '''
        Configure git-fat for usage: variables, environment
        '''
        if not self._configured():
            print('Setting filters in .git/config')
            gitconfig_set('filter.fat.clean', 'git-fat filter-clean %f')
            gitconfig_set('filter.fat.smudge', 'git-fat filter-smudge %f')
            print('Creating .git/fat/objects')
            mkdir_p(self.objdir)
            print('Initialized git-fat')

    def _configured(self):
        '''
        Returns true if git-fat is already configured
        '''
        reqs = os.path.isdir(self.objdir)
        filters = gitconfig_get('filter.fat.clean') and gitconfig_get('filter.fat.smudge')
        return filters and reqs

    def _encode(self, digest: str, size: int) -> bytes:
        '''
        Produce representation of file to be stored in repository. 20 characters can hold 64-bit integers.
        '''
        return self._format.format(digest=digest, size=size).encode('ascii')

    def _decode(self, stream: IO[bytes]) -> Tuple[Iterable[bytes], bool]:
        '''
        Returns iterator and True if stream is git-fat object
        '''
        stream_iter = iter(readblocks(stream))
        # Read block for check raises StopIteration if file is zero length
        try:
            block = next(stream_iter)
        except StopIteration:
            return stream_iter, False

        def prepend(blk: bytes, iterator: Iterator[bytes]) -> Iterable[bytes]:
            yield blk
            yield from iterator

        # Put block back
        ret = prepend(block, stream_iter)

        if block.startswith(self._cookie.encode('ascii')):
            if len(block) != self._magiclen:  # Sanity check
                warnings.warn('Found file with cookie but without magiclen')
                return ret, False
            return ret, True
        return ret, False

    def _get_digest(self, stream: IO[bytes]) -> str:
        '''
        Returns digest if stream is fatfile placeholder or '' if not
        '''
        # DONT EVER CALL THIS FUNCTION FROM FILTERS, IT DISCARDS THE FIRST
        # BLOCK OF THE INPUT STREAM.  IT IS ONLY MEANT TO CHECK THE STATUS
        # OF A FILE IN THE TREE
        stream_iter, fatfile = self._decode(stream)

        if fatfile:
            block = next(iter(stream_iter))  # read the first block
            digest = block.split()[2]
            return digest.decode('ascii')

        return ''

    def _cached_objects(self) -> FrozenSet[str]:
        '''
        Returns a set of all the cached objects
        '''
        return frozenset(os.listdir(self.objdir))

    def _referenced_objects(self, **kwargs: Any) -> FrozenSet[str]:
        '''
        Return just the hashes of the files that are referenced in the repository
        '''
        objs_dict = self._managed_files(**kwargs)
        return frozenset(objs_dict)

    def _rev_list(self) -> Iterable[Tuple[str, str, str]]:
        '''
        Generator for objects in rev. Returns (hash, type, size) tuple.
        '''

        rev = self.rev or 'HEAD'
        # full_history implies --all
        args = ['--all'] if self.full_history else ['--no-walk', rev]

        # Get all the git objects in the current revision and in history if --all is specified
        revlist = git('rev-list --objects'.split() + args, stdout=sub.PIPE)
        # Grab only the first column.  Tried doing this in python but because of the way that
        # subprocess.PIPE buffering works, I was running into memory issues with larger repositories
        # plugging pipes to other subprocesses appears to not have the memory buffer issue
        if platform.system() == "Windows":
            # Windows installer ships its own awk tool
            awk_tool = 'git-fat_gawk.exe'
        else:
            awk_tool = 'awk'

        with sub.Popen([awk_tool, '{print $1}'], stdin=revlist.stdout, stdout=sub.PIPE) as awk:
            # Read the objects and print <sha> <type> <size>
            catfile = git('cat-file --batch-check'.split(), stdin=awk.stdout, stdout=sub.PIPE)

        for line in cast(IO[str], catfile.stdout):
            objhash, objtype, size = line.split()
            yield objhash, objtype, size

        catfile.wait()

    def _find_paths(self, hashes):
        '''
        Takes a list of git object hashes and generates hash,path tuples
        '''
        rev = self.rev or 'HEAD'
        # full_history implies --all
        args = ['--all'] if self.full_history else ['--no-walk', rev]

        revlist = git('rev-list --objects'.split() + args, stdout=sub.PIPE)
        for line in revlist.stdout:
            hashobj = line.strip()
            # Revlist prints all objects (commits, trees, blobs) but blobs have the file path
            # next to the git objecthash
            # Handle files with spaces
            hashobj, _, filename = hashobj.partition(' ')
            if filename:
                # If the object is one we're managing
                if hashobj in hashes:
                    yield hashobj, filename

        revlist.wait()

    def _managed_files(self, **unused_kwargs: Any) -> Dict[str, str]:
        revlistgen = self._rev_list()
        # Find any objects that are git-fat placeholders which are tracked in the repository
        managed: Dict[str, str] = {}
        for objhash, objtype, size in revlistgen:
            # files are of blob type
            if objtype == 'blob' and int(size) == self._magiclen:
                # Read the actual file contents
                readfile = git(['cat-file', '-p', objhash], stdout=sub.PIPE)
                digest = self._get_digest(cast(io.TextIOWrapper, readfile.stdout).buffer)
                if digest:
                    managed[objhash] = digest

        # go through rev-list again to get the filenames
        # Again, I tried avoiding making another call to rev-list by caching the
        # filenames above, but was running into the memory buffer issue
        # Instead we just make another call to rev-list.  Takes more time, but still
        # only takes 5 seconds to traverse the entire history of a 22k commit repo
        filedict = dict(self._find_paths(managed.keys()))

        # return a dict(git-fat hash -> filename)
        # git's objhash are the keys in `managed` and `filedict`
        ret = dict((j, filedict[i]) for i, j in managed.items())
        return ret

    def _orphan_files(self, patterns: Optional[List[str]] = None) -> Iterable[Tuple[str, str]]:
        '''
        generator for placeholders in working tree that match pattern
        '''
        patterns = patterns or []
        # Null-terminated for proper file name handling (spaces)
        out = sub.check_output(['git', 'ls-files', '-z'] + patterns)
        for fname_enc in out.split(b'\x00')[:-1]:
            # TODO: Properly speaking should use the correct locale encoding; but
            # assume UTF-8 because we are not in Japan (yet...)
            fname = fname_enc.decode('utf-8')
            if not os.path.exists(fname):
                continue
            st = os.lstat(fname)
            if st.st_size != self._magiclen or os.path.islink(fname):
                continue
            with open(fname, "rb") as f:
                digest = self._get_digest(f)

            if digest:
                yield (digest, fname)

    def _filter_smudge(self, instream: IO[bytes], outstream: IO[bytes]) -> None:
        '''
        The smudge filter runs whenever a file is being checked out into the working copy of the tree
        instream is sys.stdin and outstream is sys.stdout when it is called by git
        '''
        blockiter, fatfile = self._decode(instream)
        if fatfile:
            block = next(iter(blockiter))  # read the first block
            digest = block.split()[2].decode('ascii')
            objfile = os.path.join(self.objdir, digest)
            try:
                with open(objfile, "rb") as f:
                    cat(f, outstream)
                logger.info(f'git-fat filter-smudge: restoring from {objfile}')
            except IOError:
                logger.info(f'git-fat filter-smudge: fat object not found in cache {objfile}')
                outstream.write(block)
        else:
            logger.info('git-fat filter-smudge: not a managed file')
            cat_iter(blockiter, sys.stdout.buffer)

    def _filter_clean(self, instream: IO[bytes], outstream: IO[bytes]) -> None:
        '''
        The clean filter runs when a file is added to the index. It gets the "smudged" (working copy)
        version of the file on stdin and produces the "clean" (repository) version on stdout.
        '''

        blockiter, is_placeholder = self._decode(instream)

        if is_placeholder:
            # This must be cat_iter, not cat because we already read from instream
            cat_iter(blockiter, outstream)
            return

        # make temporary file for writing
        fd, tmpname = tempfile.mkstemp(dir=self.objdir)
        tmpstream = os.fdopen(fd, 'wb')

        # Hash the input, write to temp file
        digest, size = hash_stream(blockiter, tmpstream)
        tmpstream.close()

        objfile = os.path.join(self.objdir, digest)

        if os.path.exists(objfile):
            logger.info(f'git-fat filter-clean: cached file already exists {objfile}')
            # Remove temp file
            delete_file(tmpname)
        else:
            # Set permissions for the new file using the current umask
            os.chmod(tmpname, int('444', 8) & ~umask())
            # Rename temp file
            move_file(tmpname, objfile)
            logger.info(f'git-fat filter-clean: caching to {objfile}')

        # Write placeholder to index
        outstream.write(self._encode(digest, size))

    def filter_clean(self, cur_file: str, **unused_kwargs: Any) -> None:
        '''
        Public command to do the clean (should only be called by git)
        '''
        logger.debug(f'CLEAN: cur_file={cur_file}, unused_kwargs={unused_kwargs}')

        if cur_file and not self.can_clean_file(cur_file):
            logger.info(f'Not adding {cur_file}. It is not a new file and is not '
                        f'managed by git-fat')
            # Git needs something, so we cat stdin to stdout
            cat(sys.stdin.buffer, sys.stdout.buffer)
        else:  # We clean the file
            if cur_file:
                logger.info(f'Adding {cur_file}')
            self._filter_clean(sys.stdin.buffer, sys.stdout.buffer)

    def filter_smudge(self, **unused_kwargs: Any) -> None:
        '''
        Public command to do the smudge (should only be called by git)
        '''
        logger.debug(f"SMUDGE: unused_kwargs={unused_kwargs}")
        self._filter_smudge(sys.stdin.buffer, sys.stdout.buffer)

    def find(self, size: int, **unused_kwargs: Any) -> None:
        '''
        Find any files over size threshold in the repository.
        '''
        revlistgen = self._rev_list()
        # Find any objects that are git-fat placeholders which are tracked in the repository
        objsizedict = {}
        for objhash, objtype, objsize in revlistgen:
            # files are of blob type
            if objtype == 'blob' and int(objsize) > size:
                objsizedict[objhash] = objsize
        for objhash, objpath in self._find_paths(objsizedict.keys()):
            print(objhash, objsizedict[objhash], objpath)

    def _parse_ls_files(self, line: str) -> Tuple[str, str, str, str]:
        mode, _, tail = line.partition(' ')
        blobhash, _, tail = tail.partition(' ')
        stageno, _, tail = tail.partition('\t')
        filename = tail.strip()
        return mode, blobhash, stageno, filename

    def _get_old_gitattributes(self) -> Tuple[List[str], str, str]:
        """ Get the last .gitattributes file in HEAD, and return it """
        ls_ga = git('ls-files -s .gitattributes'.split(), stdout=sub.PIPE)
        lsout = cast(IO[str], ls_ga.stdout).read().strip()
        ls_ga.wait()
        if lsout:  # Always try to get the old gitattributes
            ga_mode, ga_hash, ga_stno, _ = self._parse_ls_files(lsout)
            ga_cat = git(f'cat-file blob {ga_hash}'.split(), stdout=sub.PIPE)
            old_ga = cast(IO[str], ga_cat.stdout).read().splitlines()
            ga_cat.wait()
        else:
            ga_mode, ga_stno, old_ga = '100644', '0', []
        return old_ga, ga_mode, ga_stno

    def _update_index(self, uip, mode, content, stageno, filename):
        fmt = '{0} {1} {2}\t{3}\n'
        uip.stdin.write(fmt.format(mode, content, stageno, filename))

    def _add_gitattributes(self, newfiles, unused_update_index):
        """ Find the previous gitattributes file, and append to it """

        old_ga, ga_mode, ga_stno = self._get_old_gitattributes()
        ga_hashobj = git('hash-object -w --stdin'.split(), stdin=sub.PIPE,
                         stdout=sub.PIPE)
        # Add lines to the .gitattributes file
        new_ga = old_ga + [f'{f} filter=fat -text' for f in newfiles]
        stdout, _ = ga_hashobj.communicate('\n'.join(new_ga) + '\n')
        return ga_mode, stdout.strip(), ga_stno, '.gitattributes'

    def _process_index_filter_line(self, line: str, workdir: str,
                                   excludes: Container[str]) -> Optional[Tuple[str, str, str, str]]:

        mode, blobhash, stageno, filename = self._parse_ls_files(line)

        if filename not in excludes or mode == "120000":
            return None
        # Save file to update .gitattributes
        cleanedobj_hash = os.path.join(workdir, blobhash)
        # if it hasn't already been cleaned
        if not os.path.exists(cleanedobj_hash):
            catfile = git(f'cat-file blob {blobhash}'.split(), stdout=sub.PIPE)
            hashobj = git('hash-object -w --stdin'.split(), stdin=sub.PIPE, stdout=sub.PIPE)
            self._filter_clean(cast(io.TextIOWrapper, catfile.stdout).buffer,
                               cast(io.TextIOWrapper, hashobj.stdin).buffer)
            cast(IO[str], hashobj.stdin).close()
            objhash = cast(IO[str], hashobj.stdout).read().strip()
            catfile.wait()
            hashobj.wait()
            with open(cleanedobj_hash, 'w', encoding='utf-8') as cleaned:
                cleaned.write(objhash + '\n')
        else:
            with open(cleanedobj_hash, 'r', encoding='utf-8') as cleaned:
                objhash = cleaned.read().strip()
        return mode, objhash, stageno, filename

    def index_filter(self, filelist: str, add_gitattributes: bool = True,
                     **unused_kwargs: Any) -> None:
        gitdir = sub.check_output('git rev-parse --git-dir'.split(),
                                  encoding='utf-8').strip()
        workdir = os.path.join(gitdir, 'fat', 'index-filter')
        mkdir_p(workdir)

        with open(filelist, 'r', encoding='utf-8') as excludes:
            files_to_exclude = frozenset(excludes.read().splitlines())

        ls_files = git('ls-files -s'.split(), stdout=sub.PIPE)
        uip = git('update-index --index-info'.split(), stdin=sub.PIPE)

        newfiles = []
        for line in cast(IO[str], ls_files.stdout):
            newfile = self._process_index_filter_line(line, workdir, files_to_exclude)
            if newfile:
                self._update_index(uip, *newfile)
                # The filename is in the last position
                newfiles.append(newfile[-1])

        if add_gitattributes:
            # Add the files to the gitattributes file and update the index
            attrs = self._add_gitattributes(newfiles, add_gitattributes)
            self._update_index(uip, *attrs)

        ls_files.wait()
        cast(IO[str], uip.stdin).close()
        uip.wait()

    def list_files(self, **kwargs):
        '''
        Command to list the files by fat-digest -> gitroot relative path
        '''
        managed = self._managed_files(**kwargs)
        for f in managed.keys():
            print(f, managed.get(f))

    def _remove_orphan_file(self, fname):
        # The output of our smudge filter depends on the existence of
        # the file in .git/fat/objects, but git caches the file stat
        # from the previous time the file was smudged, therefore it
        # won't try to re-smudge. There's no git command to specifically
        # invalidate the index cache so we have two options:
        # Change the file stat mtime or change the file size. However, since
        # the file mtime only has a granularity of 1s, if we're doing a pull
        # right after a clone or checkout, it's possible that the modified
        # time will be the same as in the index. Git knows this can happen
        # so git checks the file size if the modified time is the same.
        # The easiest way around this is just to remove the file we want
        # to replace (since it's an orphan, it should be a placeholder)
        with open(fname, 'rb') as f:
            recheck_digest = self._get_digest(f)  # One last sanity check
        if recheck_digest:
            delete_file(fname)

    def checkout_all_index(self, show_orphans=False, **unused_kwargs):
        '''
        Checkout all files from index when restoring many binaries, to enhance the performance.
        Need the working directory to be clean.
        '''
        # avoid unstaged changed being overwritten
        if sub.check_output(["git", "ls-files", "-m"]):
            print('You have unstaged changes in working directory')
            print('please use "git add <file>..." to stage those changes'
                  ' or use "git checkout -- <file>..." to discard changes')
            sys.exit(1)

        for digest, fname in self._orphan_files():
            objpath = os.path.join(self.objdir, digest)
            if os.access(objpath, os.R_OK):
                print(f'Will restore {digest} -> {fname}')
                self._remove_orphan_file(fname)
            elif show_orphans:
                print(f'Data unavailable: {digest} {fname}')

        print('Restoring files ...')
        # This re-smudge is essentially a copy that restores permissions.
        sub.check_call(['git', 'checkout-index', '--index', '--force', '--all'])

    def checkout(self, show_orphans=False, **unused_kwargs):
        '''
        Update any stale files in the present working tree
        '''
        for digest, fname in self._orphan_files():
            objpath = os.path.join(self.objdir, digest)
            if os.access(objpath, os.R_OK):
                print(f'Restoring {digest} -> {fname}')
                self._remove_orphan_file(fname)
                # This re-smudge is essentially a copy that restores permissions.
                sub.check_call(['git', 'checkout-index', '--index', '--force', fname])
            elif show_orphans:
                print(f'Data unavailable: {digest} {fname}')

    def can_clean_file(self, filename: str) -> bool:
        '''
        Checks to see if the current file exists in the local repo before filter-clean
        This method prevents fat from hijacking glob matches that are old
        '''
        # If the file doesn't exist in the immediately previous revision, add it
        showfile = git(['show', f'HEAD:{filename}'], stdout=sub.PIPE, stderr=sub.PIPE)

        blockiter, is_fatfile = self._decode(cast(io.TextIOWrapper, showfile.stdout).buffer)

        # Flush the buffers to prevent deadlock from wait()
        # Caused when stdout from showfile is a large binary file and can't be fully buffered
        # I haven't figured out a way to avoid this unfortunately
        for _ in blockiter:
            continue

        if showfile.wait() or is_fatfile:
            # The file didn't exist in the repository
            # The file was a fatfile (which may have changed)
            return True

        # File exists but is not a fatfile, don't add it
        return False

    def pull(self, patterns: Optional[List[str]] = None, **kwargs: Any) -> None:
        """ Get orphans, call backend pull """
        cached_objs = self._cached_objects()

        # TODO: Why use _orphan _and_ _referenced here?
        if patterns:
            # filter the working tree by a pattern
            orphan_files = self._orphan_files(patterns=patterns)
            files = frozenset(digest for digest, fname in orphan_files) - cached_objs
        else:
            # default pull any object referenced but not stored
            files = self._referenced_objects(**kwargs) - cached_objs

        logger.debug(f'PULL: patterns={patterns}, kwargs={kwargs}, len(files)={len(files)}')

        if not self.backend.pull_files(files):
            sys.exit(1)
        # Make sure they're up to date
        if kwargs.pop("many_binaries", False):
            print('in accelerating mode')
            self.checkout_all_index()
        else:
            self.checkout()

    def push(self, unused_pattern: Any = None, **kwargs: Any) -> None:
        # We only want the intersection of the referenced files and ones we have cached
        # Prevents file doesn't exist errors, while saving on bw by default (_referenced only
        # checks HEAD for files)
        files = self._referenced_objects(**kwargs) & self._cached_objects()
        logger.debug(f'PUSH: unused_pattern={unused_pattern}, kwargs={kwargs}, len(files)={len(files)}')
        if not self.backend.push_files(files):
            sys.exit(1)

    def _status(self, **kwargs: Any) -> Tuple[FrozenSet[str], FrozenSet[str]]:
        '''
        Helper function that returns the oprhans and stale files
        '''
        catalog = self._cached_objects()
        referenced = self._referenced_objects(**kwargs)
        stale = catalog - referenced
        orphans = referenced - catalog
        return stale, orphans

    def status(self, **kwargs: Any) -> None:
        '''
        Show orphan (in tree, but not in cache) and stale (in cache, but not in tree) objects, if any.
        '''
        stale, orphans = self._status(**kwargs)
        if orphans:
            print('Orphan objects:')
            for orph in orphans:
                print('\t' + orph)
        if stale:
            print('Stale objects:')
            for g in stale:
                print('\t' + g)


def _get_options(config: cfgparser.ConfigParser, backend: str, cfg_file_path: str) -> Dict[str, str]:
    """ returns the options for a backend in dictionary form """
    try:
        opts = dict(config.items(backend))
    except cfgparser.NoSectionError as exc:
        err = f'No section found in {cfg_file_path} for backend {backend}'
        raise RuntimeError(err) from exc
    return opts


def _read_config(cfg_file_path: str) -> cfgparser.ConfigParser:
    config = cfgparser.ConfigParser()

    if not os.path.exists(cfg_file_path):
        # Can't continue, but this isn't unusual
        logger.warning(f'This does not appear to be a repository managed by git-fat. '
                       f'Missing configfile at: {cfg_file_path}')
        sys.exit(0)

    try:
        config.read(cfg_file_path)
    except cfgparser.Error as exc:  # TODO: figure out what to catch here
        raise RuntimeError(f'Error reading or parsing configfile: {cfg_file_path}') from exc

    return config


def _parse_config(backend: Optional[str] = None, cfg_file_path: Optional[str] = None) -> BackendInterface:
    """ Parse the given config file and return the backend instance """
    cfg_file_path = _config_path(path=cfg_file_path)
    assert cfg_file_path
    config = _read_config(cfg_file_path)
    if backend is None:
        try:
            backends = config.sections()
        except cfgparser.Error as exc:
            raise RuntimeError(f'Error reading or parsing configfile: {cfg_file_path}') from exc
        if not backends:  # e.g. empty file
            raise RuntimeError(f'No backends configured in config: {cfg_file_path}')
        backend = backends[0]

    opts = _get_options(config, backend, cfg_file_path)
    base_dir = _obj_dir()

    try:
        Backend = BACKEND_MAP[backend]
    except IndexError as exc:
        raise RuntimeError(f'Unknown backend specified: {backend}') from exc

    return Backend(base_dir, **opts)


def run(backend, **kwargs):
    make_sys_streams_binary()
    name = kwargs.pop('func')
    full_history = kwargs.pop('full_history')
    gitfat = GitFat(backend, full_history=full_history)
    fn = name.replace("-", "_")
    if not hasattr(gitfat, fn):
        raise Exception("Unknown function called")
    getattr(gitfat, fn)(**kwargs)


def _configure_logging(log_level):
    if GIT_FAT_LOG_LEVEL:
        log_level = GIT_FAT_LOG_LEVEL
    if GIT_FAT_LOG_FILE:
        file_handler = _logging.FileHandler(GIT_FAT_LOG_FILE)
        file_handler.setLevel(log_level)
        formatter = _logging.Formatter(
            '%(levelname)s:%(filename)s:%(funcName)s:%(lineno)d: %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    logger.setLevel(log_level)


def _load_backend(kwargs: Dict[str, Any]) -> Optional[BackendInterface]:
    needs_backend = ('pull', 'push')
    backend_opt = kwargs.pop('backend', None)
    config_file = kwargs.pop('config_file', None)
    backend = None
    if kwargs['func'] == 'pull':
        # since pull can be of the form pull [backend] [patterns], we need to check
        # the first argument and insert into file patterns if it's not a backend
        # this means you can't use a file pattern which is an exact match with
        # a backend name (e.g. you can't have a file named copy, rsync, http, etc)
        if backend_opt and backend_opt not in BACKEND_MAP:
            kwargs['patterns'].insert(0, backend_opt)
            backend_opt = None
    if kwargs['func'] in needs_backend:
        backend = _parse_config(backend=backend_opt, cfg_file_path=config_file)
    return backend


def main() -> None:

    parser = argparse.ArgumentParser(
        argument_default=argparse.SUPPRESS,
        description='A tool for managing large binary files in git repositories.')
    subparser = parser.add_subparsers()

    # Global options
    parser.add_argument(
        '-a', "--full-history", dest='full_history', action='store_true', default=False,
        help='Look for git-fat placeholder files in the entire history instead of just the working copy')
    parser.add_argument(
        '-v', "--verbose", dest='verbose', action='store_true',
        help='Get verbose output about what git-fat is doing')
    parser.add_argument(
        '-d', "--debug", dest='debug', action='store_true',
        help='Get debugging output about what git-fat is doing')
    parser.add_argument(
        '-c', "--config", dest='config_file', type=str,
        help='Specify which config file to use (defaults to .gitfat)')

    # redundant function for legacy api; config gets called every time.
    # (assuming if user is calling git-fat they want it configured)
    # plus people like running init when setting things up d(^_^)b
    sp = subparser.add_parser('init', help='Initialize git-fat')
    sp.set_defaults(func="configure")

    sp = subparser.add_parser('filter-clean', help="Internal function used by git")
    sp.add_argument("cur_file", nargs="?")
    sp.set_defaults(func='filter_clean')

    sp = subparser.add_parser('filter-smudge', help="Internal function used by git")
    sp.add_argument("cur_file", nargs="?")
    sp.set_defaults(func='filter_smudge')

    sp = subparser.add_parser('push', help='push cache to remote git-fat server')
    sp.add_argument("backend", nargs="?", help='pull using given backend')
    sp.set_defaults(func='push')

    sp = subparser.add_parser('pull', help='pull fatfiles from remote git-fat server')
    sp.add_argument("backend", nargs="?", help='pull using given backend')
    sp.add_argument("--many-binaries", dest='many_binaries', action='store_true',
                    help='accelerate pulling a repository which contains many binaries')
    sp.add_argument("patterns", nargs="*", help='files or file patterns to pull')
    sp.set_defaults(func='pull')

    sp = subparser.add_parser('checkout', help='resmudge all orphan objects')
    sp.set_defaults(func='checkout')

    sp = subparser.add_parser('find', help='find all objects over [size]')
    sp.add_argument("size", type=int, help='threshold size in bytes')
    sp.set_defaults(func='find')

    sp = subparser.add_parser('status', help='print orphan and stale objects')
    sp.set_defaults(func='status')

    sp = subparser.add_parser('list', help='list all files managed by git-fat')
    sp.set_defaults(func='list_files')

    # Legacy function to preserve backwards compatability
    sp = subparser.add_parser('pull-http', help="Deprecated, use `pull http` (no dash) instead")
    sp.set_defaults(func='pull', backend='http')

    sp = subparser.add_parser('index-filter', help='git fat index-filter for filter-branch')
    sp.add_argument('filelist', help='file containing all files to import to git-fat')
    sp.add_argument(
        '-x', dest='add_gitattributes',
        help='prevent adding excluded to .gitattributes', action='store_false')
    sp.set_defaults(func='index_filter')

    if len(sys.argv) > 1 and sys.argv[1] in [c + 'version' for c in ['', '-', '--']]:
        print(__version__)
        sys.exit(0)

    args = parser.parse_args()
    kwargs = dict(vars(args))

    if kwargs.pop('debug', None):
        log_level = _logging.DEBUG
    elif kwargs.pop('verbose', None):
        log_level = _logging.INFO
    else:
        log_level = _logging.WARNING
    _configure_logging(log_level)

    try:
        backend = _load_backend(kwargs)  # load_backend mutates kwargs
        run(backend, **kwargs)
    except RuntimeError as err:
        logger.error(str(err))
        sys.exit(1)
    except Exception:
        if cur_file := kwargs.get('cur_file'):
            logger.error(f'processing file: {cur_file}')
        raise


if __name__ == '__main__':
    main()

__all__ = ['__version__', 'main', 'GitFat']
