"""Validate candidate joins without silently replacing keys by row numbers."""
def validate_ids(frame, key, name):
    if key not in frame:
        raise ValueError(f'{name}: missing {key!r}')
    if frame[key].isna().any() or frame[key].astype(str).duplicated().any():
        raise ValueError(f'{name}: {key!r} must be nonmissing and unique')


def validate_label_join(raw, labels, key):
    validate_ids(raw,key,'structures')
    validate_ids(labels,key,'labels')
    left=raw.assign(_key=raw[key].astype(str)).set_index('_key')
    right=labels.assign(_key=labels[key].astype(str)).set_index('_key')
    if not set(right.index)<=set(left.index):
        raise ValueError('Labels contain candidate IDs absent from the structure file')
    # Candidate IDs may repeat between seeds. Check their final CIFs too.
    if 'cif_str' in left and 'cif_str' in right:
        normalize=lambda x:str(x).replace('\\n','\n').strip()
        for identifier in right.index:
            if normalize(left.at[identifier,'cif_str']) != normalize(right.at[identifier,'cif_str']):
                raise ValueError(f'CIF differs for {identifier}; check seed/run and ID column')
