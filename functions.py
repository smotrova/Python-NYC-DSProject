
# =============================================================================
# # function that capitalize each word in a string
# =============================================================================

def convert_str(s):
    
    '''
        Input: string
        Output: capitilized string
    
    '''
    
    words_list = []
    
    for word in s.split():
        words_list.append(word.capitalize())
        
    return ' '.join(words_list)