import wikipediaapi #imports wikipedia api

wiki_wiki = wikipediaapi.Wikipedia('en', extract_format=wikipediaapi.ExtractFormat.WIKI) #sets language and extraction format


page_py = wiki_wiki.page('Van Halen') #sets wikipedia page
print("Page - Exists: %s" % page_py.exists()) #checks if it exists

print("Page - Title: %s" % page_py.title) #prints page title

print("Page - Summary: %s" % page_py.summary[0:400]) #prints first 400 characters about the topic

print(page_py) #prints entire page
