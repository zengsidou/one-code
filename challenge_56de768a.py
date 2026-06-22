import re

def fix_html(html):
    tags = re.findall(r'<(/?)([a-z]+)[^>]*>', html)
    stack = []
    for closing, tag in tags:
        if closing:
            while stack and stack[-1] != tag:
                html += '</' + stack.pop() + '>'
            if stack:
                stack.pop()
        else:
            if tag not in ('img', 'br', 'hr', 'input'):
                stack.append(tag)
    while stack:
        html += '</' + stack.pop() + '>'
    return html

test = '<div><p>hello<img src="image.png"></div>'
print(fix_html(test))