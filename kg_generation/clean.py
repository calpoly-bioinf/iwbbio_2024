# Functions to clean text
import string
import json
import re

import spacy
import crosslingual_coreference
import numpy as np

import openai
# TODO: not hard code
openai.api_key = open("/mnt/clbp/.openai_api_key.txt").read().strip()

nlp = spacy.load("en_core_web_sm")

def is_paragraph_beginning(text):
    # Define criteria for identifying the beginning of a paragraph
    doc = nlp(text)
    count = 0
    for sent in doc:
        if "VerbForm=Fin" in sent.morph:
            count += 1
    # If none of the conditions are met, it may not be the beginning of a paragraph
    if count > 1:
        return True
    return False

def get_raw_pdf_to_json_func2():
    
    def process(infile):
        content = open(infile).read()
        outfile = infile.replace(".pdf.txt",".pdf.json")
        title = infile.replace(".pdf.txt","")
        
        new_sentences = []
        content = re.sub(r"\.\n",".\n\n",content)

        #content = content.replace(r".\n",".\n\n")
        paragraphs = content.split("\n\n")
        print(len(paragraphs))
        #print("\n\n".join(paragraphs))
        #paragraphs = content.split("\n\n")
        printable = set(string.printable)
        min_num_words_in_sentences = 4
        new_paragraphs = []
        for p in paragraphs:
            new_p = ''.join(filter(lambda x: x in printable, p)).strip()
            new_p = new_p.strip()
            if is_paragraph_beginning(new_p.replace("\n"," ")):
                new_paragraphs.append(new_p.replace("\n"," ").strip().replace("  "," "))
            #else:
            #    import pdb; pdb.set_trace()

        print(len(new_paragraphs))
        new_paragraphs_joined = []
        last_p = None
        for p in new_paragraphs:
            if last_p is None or last_p.endswith("."):
                new_paragraphs_joined.append(p)
            else:
                new_paragraphs_joined[-1] = (new_paragraphs_joined[-1] + " " + p).replace("  "," ")
            last_p = new_paragraphs_joined[-1]

        sections = {}
        sections['title'] = title
        sections['contents'] = "\n\n".join(new_paragraphs_joined)
        open(outfile,"w").write(json.dumps(sections))

        return new_paragraphs_joined
    return process

def get_related_content_func1():
    def process(infile,outfile,hypothesis):
        sections = json.loads(open(infile).read())
        content = sections["contents"]
        
        paragraphs = content.split("\n\n")

        PROMPT = """
        Your job is to read the following paper, and tell me about whether this paper contains information relevant to the following hypothesis:
        """+hypothesis+"""
        Your response should be yes or no with no extra information.
        """
        
        max_tokens = 4097
        relevant_paragraphs = []
        non_relevant_paragraphs = []
        for p in paragraphs:
            messages=[
                  {"role": "system", "content": PROMPT},
                  {"role": "user", "content": p }
            ]
            finished = False
            c = 1
            while not finished:
                if c > 3:
                    break
                if c > 1:
                    print(f"Trying openai for {c} time")
                try:
                    response = openai.ChatCompletion.create(
                      model="gpt-3.5-turbo",
                      messages=messages,
                      temperature=0,
                      top_p=1,
                      frequency_penalty=0,
                      presence_penalty=0
                    )
                    finished = True
                except Exception as e:
                    print(e)
                c += 1

            if finished:
                response = response['choices'][0]['message']['content'].strip()
                if response.lower() == "no":
                    non_relevant_paragraphs.append(p)
                else:
                    relevant_paragraphs.append(p)
            else:
                relevant_paragraphs.append(p) # assume relevant
        
        contents = "\n\n".join(relevant_paragraphs)
        removed_contents = "\n\n".join(non_relevant_paragraphs)
        
        sections['contents'] = contents
        sections['removed_contents'] = removed_contents
        open(outfile,"w").write(json.dumps(sections))

        return relevant_paragraphs,non_relevant_paragraphs
    return process

def get_raw_pdf_to_json_func1():
    def process(infile):
        content = open(infile).read()
        outfile = infile.replace(".pdf.txt",".pdf.json")
        title = infile.replace(".pdf.txt","")
        
        new_sentences = []
        paragraphs = content.split("\n\n")
        for p in paragraphs:
            new_sentences.extend(p.replace("\n"," ").replace("- ","").split("."))
        
        printable = set(string.printable)
        
        max_tokens = 4097
        batch_size = 20 # number of sentences to ask at a time
        min_num_words_in_sentences = 4
        min_len_sentence = min_num_words_in_sentences*4
        
        valid_sentences = []
        all_sentences = list(new_sentences)
        i = 0
        while i < len(all_sentences):
            gpt_prompt = f"""
        Classify whether Text is a complete sentence without typos. Return the indices classified as complete as a csv.
        """
            count = 0
            tested_sentences = []
            while count < batch_size and i < len(all_sentences) and max_tokens-len(gpt_prompt) > 500:
                sent = all_sentences[i]
                i += 1
                sent = ''.join(filter(lambda x: x in printable, sent)).strip()
                num_words = len(sent.split(" "))
                if num_words < min_num_words_in_sentences or len(sent) < min_len_sentence:
                    continue
                count2 = count + 1
                gpt_prompt += f"\nText {count2}. \"{sent}\""
                tested_sentences.append(sent+".")
                count += 1
        
            gpt_prompt += "\n\nSentences:"
            #print(gpt_prompt)
            try:
                response = openai.Completion.create(
                  engine="text-davinci-003",
                  prompt=gpt_prompt,
                  temperature=0,
                  max_tokens=max_tokens-len(gpt_prompt),
                  top_p=1.0,
                  frequency_penalty=0.0,
                  presence_penalty=0.0
                )
            
                for j,response in enumerate(response['choices']):
                    text = response['text'].strip()
                    ixs = text.split("\n")[0].replace("Text","").replace(" and ",",").replace("and","").replace(".","").split(",")
                    cleaned_ixs = []
                    for ix in ixs:
                        try:
                            if "-" in ix:
                                start,end = ix.split("-")
                                for j in range(int(start)-1,int(end)):
                                    cleaned_ixs.append(j)
                            else:
                                cleaned_ixs.append(int(ix)-1)
                        except:
                            print("Error somewhere in",text)
                    valid_sentences.extend(list(np.array(tested_sentences)[cleaned_ixs]))
            except:
                print("Error")
            #print("\n".join(valid_sentences))
            #print(gpt_prompt+text)
            #if text.startswith("Yes"):
            #    valid_sentences.append((sent+".").strip())
        
        
            print(i+1,"out of",len(all_sentences),"sentences or",int(100*(i+1)/len(all_sentences)),"percent.")
            #if i % 50 == 0:
            #    print('Sleeping')
            #    time.sleep(60)
        
        contents = "\n".join(valid_sentences)
        
        sections = {}
        sections['title'] = title
        sections['contents'] = contents
        open(outfile,"w").write(json.dumps(sections))
    return process

def get_coref_func1(DEVICE=0,chunk_size=2500,chunk_overlap=2,model='en_core_web_lg'):
    DEVICE = 0 # Number of the GPU, -1 if want to use CPU
    # Add coreference resolution model
    coref = spacy.load(model, disable=['ner', 'tagger', 'parser', 'attribute_ruler', 'lemmatizer'])
    coref.add_pipe(
        "xx_coref", config={"chunk_size": chunk_size, "chunk_overlap": chunk_overlap, "device": DEVICE})

    # TODO: Need to figure out how to identify the parts of the text that were replaced and mark those with [] for debugging
    def resolveText2Text(text):
        coref_text = coref(text)._.resolved_text
        return coref_text,text,coref_text    
        
    return resolveText2Text

"""
  model="gpt-3.5-turbo",
  max_tokens=256,
  messages=[
        {'role': 'user', 'content': 'You are an expert in English Grammar. Rewrite a given sentence such that each sentence \
has one subject and one predicate. Keep the meaning the same, write in active voice, and try to make the sentences independent and unique.'},
        {'role': 'user', 'content': 'here are the sentences list:
              1:sentence1.
              2:sentence2.
              3:sentence1.
              4:sentence2.
              5:sentence1.
              ....
              100:sentence2.'},
       {'role': 'user', 'content': 'Please process each sentence. Your response should take relevant details from the background, and the response. The output should only contain the sentence index number and the new sentences you wrote.'}
    ]
)
"""

# Returns a function that rewrites all sentences in a paragraph
def get_rewrite_func2():
    def rewrite(paragraph,verbose=False,max_sentences=100):
        '''Rewrite a paragraph using only simple sentences:
        Input: paragraph
        Output: Paragraph'''
        final_sentences = []
        messages=[
            {'role': 'user', 'content': 'You are an expert in English Grammar. You know the difference between simple, compound, complex, and compound-complex sentences. I want you to tell me if a sentence is simple, compound, complex, or compound-complex.'}]
        sent_part = ""
        doc = nlp(paragraph)
        sents = list(doc.sents)
        if len(sents) > max_sentences:
            return []
        for i,sent in enumerate(sents):
            j = i+1
            sent = str(sent)
            if sent.strip() == "":
                continue
            sent_part += f"\n{j}:{sent}."
        messages.extend([{'role': 'user', 'content': 'Here are the sentences list:'+sent_part},
                         {'role': 'user', 'content': 'Tell me if each sentence is simple, compound, complex, or compound-complex. Your response should take relevant details from the background, and the response. The output should only contain the sentence index number and one of the following: simple, compound, complex, or compound-complex.'}])
       
        finished = False
        c = 1
        while not finished:
            if c > 3:
                break
            if c > 1:
                print(f"Trying openai for {c} time")
            try:
                response = openai.ChatCompletion.create(
                  model="gpt-3.5-turbo",
                  messages=messages,
                  temperature=0,
                  max_tokens=1024,
                  top_p=1,
                  frequency_penalty=0,
                  presence_penalty=0
                )
                finished = True
            except Exception as e:
                print(e)
            c += 1
        sentences = response['choices'][0]['message']['content'].split("\n")
        if sentences[0].strip() == "None":
            return ""
        try:
            classifications = [s.split(":")[1].strip() for s in sentences]
        except:
            return ""
        if len(classifications) == 0 or len(classifications) != len(sents):
            return ""

        for i,sent in enumerate(sents):
            sent = str(sent)
            if sent.strip() == "":
                continue
            if classifications[i] == 'simple':
                final_sentences.append(sent)
                continue
            classification = classifications[i]
            messages=[
                {'role': 'system', 'content': 'You are an expert in English Grammar. You know the difference between simple, compound, complex, and compound-complex sentences.'},
                {'role': 'user', 'content': f'I want you to rewrite a sentence that is {classification} into simple sentences without modifiers. Keep the meaning the same. Do not use commas. Your answer should be a numbered list.'}]
            messages.extend([{'role': 'user', 'content': sent}])
            finished = False
            c = 1
            while not finished:
                if c > 3:
                    break
                if c > 1:
                    print(f"Trying openai for {c} time")
                try:
                    response = openai.ChatCompletion.create(
                      model="gpt-3.5-turbo",
                      messages=messages,
                      temperature=0,
                      max_tokens=1024,
                      top_p=1,
                      frequency_penalty=0,
                      presence_penalty=0
                    )
                    finished = True
                except Exception as e:
                    print(e)
                c+=1
            if finished:
                sentences = response['choices'][0]['message']['content'].split("\n")
                if sentences[0].strip() == "None":
                    continue
                for s in sentences:
                    words = s.split(" ")
                    if len(words) > 1:
                        final_sentences += [" ".join(words[1:]).strip()] # Remove the number
                    else:
                        final_sentences += [s]
            else:
                final_sentences += [sent]
        return final_sentences
    return rewrite

# Returns a function that rewrites all sentences in a paragraph
def get_rewrite_func1():
    REWRITE_PROMPT = "You are an expert in English Grammar. Rewrite the given sentence such that each sentence \
has one subject and one predicate. Keep the meaning the same and write in active voice. \
Your answer should be a numbered list."
    def rewrite(paragraph,verbose=False):
        '''Rewrite a paragraph using only simple sentences:
        Input: paragraph
        Output: Paragraph'''
        final_sentences = []
        for sent in paragraph.split("."):
            if verbose:
                print(sent[:10])
            if sent.strip() == "":
                continue
            messages=[
                  {"role": "system", "content": REWRITE_PROMPT},
                  {"role": "user", "content": sent }
            ]
            finished = False
            while not finished:
                if verbose:
                    print("Trying openai")
                try:
                    response = openai.ChatCompletion.create(
                      model="gpt-3.5-turbo",
                      messages=messages,
                      temperature=0,
                      max_tokens=1024,
                      top_p=1,
                      frequency_penalty=0,
                      presence_penalty=0
                    )
                    finished = True
                except Exception as e:
                    print(e)
            sentences = response['choices'][0]['message']['content'].split("\n")
            if sentences[0].strip() == "None":
                continue
            final_sentences += [" ".join(s.split(" ")[1:]) for s in sentences] # Remove the number
        return final_sentences
    return rewrite


def get_simplify_func1():
    pass

def get_triplet_func1():
    pass

def get_triplet_func2():
    pass