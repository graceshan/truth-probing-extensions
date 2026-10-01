"""Pure Section 9 renderer shared identically by D and E; standard library only."""
CONDITION = 'and_both_following_v1'
PREFIX = 'Both of the following are true: '
EXPRESSION = '"Both of the following are true: " + first + " " + second'


def render(first, second, operator='AND'):
    if operator != 'AND':
        raise ValueError('Section 9 condition requires AND')
    if not all(isinstance(s,str) and s and s==s.strip() and s.endswith('.') for s in (first,second)):
        raise ValueError('complete exact constituent sentences required')
    return PREFIX + first + ' ' + second


def validate_spec(spec):
    if not (spec['schema_version']==1 and spec['condition_id']==CONDITION and
            spec['operator']=='AND' and spec['prefix']==PREFIX and spec['renderer']==EXPRESSION and
            spec['provenance']['section']=='9' and spec['provenance']['quotation']=='Both of the following are true: A. B.' and
            spec['held_out_from_compound_fitting'] is True and spec['held_out_from_selection'] is True and
            spec['final_evaluation_enabled'] is False and spec['P_A_frozen'] is False):
        raise ValueError('wrong frozen Section 9 specification')
