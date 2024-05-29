import cupy as cp

class InitStateConstructor(object):
    @staticmethod
    def Choice(size, choices=[0, 1], p=0.5):
        if not hasattr(p, '__len__'):
            p = [p, 1 - p]
        return cp.random.choice([cp.float32(ch) for ch in choices], (size, ) * 3, p=p)
    
    @staticmethod
    def FromLeniaRLE(str):
        DIM = 3
        DIM_DELIM = {0:'', 1:'$', 2:'%', 3:'#', 4:'@A', 5:'@B', 6:'@C', 7:'@D', 8:'@E', 9:'@F'}

        def ch2val(c):
            if c in '.b': return 0
            elif c == 'o': return 255
            elif len(c) == 1: return ord(c)-ord('A')+1
            else: return (ord(c[0])-ord('p')) * 24 + (ord(c[1])-ord('A')+25)

        def _append_stack(list1, list2, count, is_repeat=False):
            list1.append(list2)
            if count != '':
                repeated = list2 if is_repeat else []
                list1.extend([repeated] * (int(count)-1))

        def _recur_get_max_lens(dim, list1, max_lens):
            max_lens[dim] = max(max_lens[dim], len(list1))
            if dim < DIM-1:
                for list2 in list1:
                    _recur_get_max_lens(dim+1, list2, max_lens)

        def _recur_cubify(dim, list1, max_lens):
            more = max_lens[dim] - len(list1)
            if dim < DIM-1:
                list1.extend([[]] * more)
                for list2 in list1:
                    _recur_cubify(dim+1, list2, max_lens)
            else:
                list1.extend([0] * more)

        def _rle2arr(st):
            stacks = [[] for _ in range(DIM)]
            last, count = '', ''
            delims = list(DIM_DELIM.values())
            st = st.rstrip('!') + DIM_DELIM[DIM-1]
            for ch in st:
                if ch.isdigit(): count += ch
                elif ch in 'pqrstuvwxy@': last = ch
                else:
                    if last+ch not in delims:
                        _append_stack(stacks[0], ch2val(last+ch)/255, count, is_repeat=True)
                    else:
                        dim = delims.index(last+ch)
                        for d in range(dim):
                            _append_stack(stacks[d+1], stacks[d], count, is_repeat=False)
                            stacks[d] = []
                    last, count = '', ''
            A = stacks[DIM-1]
            max_lens = [0 for _ in range(DIM)]
            _recur_get_max_lens(0, A, max_lens)
            _recur_cubify(0, A, max_lens)
            return cp.asarray(A, dtype=cp.float32)
        
        return _rle2arr(str)