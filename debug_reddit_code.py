from bs4 import BeautifulSoup

def extract_data(soup):
    items = []
    
    # Find all post containers
    posts = soup.select('div[data-testid="post-container"]')
    
    for post in posts:
        item = {}
        
        # Extract title
        title_elem = post.select_one('h3, h4, [class*="title"], [data-testid*="title"]')
        item['title'] = title_elem.text.strip() if title_elem else None
        
        # Extract author
        author_elem = post.select_one('[class*="author"], [data-testid*="author"]')
        item['author'] = author_elem.text.strip() if author_elem else None
        
        # Extract upvotes
        upvotes_elem = post.select_one('[data-testid*="upvote-count"]')
        item['upvotes'] = upvotes_elem.text.strip() if upvotes_elem else None
        
        # Extract comments count
        comments_elem = post.select_one('[data-testid*="comment-count"]')
        item['comments_count'] = comments_elem.text.strip() if comments_elem else None
        
        items.append(item)
    
    return items