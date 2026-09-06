function root = parseXmlMini(txt)
%PARSEXMLMINI  Minimal XML parser (elements, attributes, text) with no Java.
%
%   root = internal.parseXmlMini(txt) returns the document element as a
%   node struct: .name (char, namespace prefix removed), .attrs (struct of
%   char values), .children (1 x n cell of nodes), .text (concatenated
%   character data, trimmed). Comments, processing instructions, DOCTYPE,
%   and CDATA markers are skipped; the five predefined entities and
%   numeric character references are decoded. Enough for MusicXML; not a
%   general validator.

    txt = char(txt);
    % Strip comments, processing instructions, and the DOCTYPE.
    txt = regexprep(txt, '<!--.*?-->', '');
    txt = regexprep(txt, '<\?.*?\?>', '');
    txt = regexprep(txt, '<!DOCTYPE[^\[>]*(\[.*?\])?\s*>', '');
    txt = strrep(strrep(txt, '<![CDATA[', ''), ']]>', '');
    % Tokens: tags and the text between them.
    toks = regexp(txt, '<[^>]+>|[^<]+', 'match');
    stack = {};
    root = [];
    for i = 1:numel(toks)
        tk = toks{i};
        if tk(1) ~= '<'
            if ~isempty(stack)
                s = strtrim(localDecode(tk));
                if ~isempty(s)
                    stack{end}.text = [stack{end}.text s];
                end
            end
            continue;
        end
        if tk(2) == '/'                                   % closing tag
            node = stack{end};
            stack(end) = [];
            if isempty(stack)
                root = node;
            else
                stack{end}.children{end + 1} = node;
            end
            continue;
        end
        selfClose = tk(end - 1) == '/';
        body = tk(2:end - 1 - selfClose);
        nameEnd = find(isspace(body), 1);
        if isempty(nameEnd)
            name = body; attrTxt = '';
        else
            name = body(1:nameEnd - 1); attrTxt = body(nameEnd + 1:end);
        end
        colon = find(name == ':', 1, 'last');
        if ~isempty(colon)
            name = name(colon + 1:end);
        end
        node = struct('name', name, 'attrs', struct(), 'children', {{}}, 'text', '');
        if ~isempty(attrTxt)
            at = regexp(attrTxt, '([\w:.-]+)\s*=\s*("([^"]*)"|''([^'']*)'')', 'tokens');
            for a = 1:numel(at)
                key = regexprep(at{a}{1}, '[^A-Za-z0-9_]', '_');
                val = at{a}{2};
                node.attrs.(key) = localDecode(val(2:end - 1));
            end
        end
        if selfClose
            if isempty(stack)
                root = node;
            else
                stack{end}.children{end + 1} = node;
            end
        else
            stack{end + 1} = node; %#ok<AGROW>
        end
    end
    if isempty(root)
        error('readScore:xml', 'Malformed MusicXML (no document element).');
    end
end


function s = localDecode(s)
    if ~any(s == '&')
        return;
    end
    s = regexprep(s, '&#x([0-9A-Fa-f]+);', '${char(hex2dec($1))}');
    s = regexprep(s, '&#([0-9]+);', '${char(str2double($1))}');
    s = strrep(s, '&lt;', '<');
    s = strrep(s, '&gt;', '>');
    s = strrep(s, '&quot;', '"');
    s = strrep(s, '&apos;', '''');
    s = strrep(s, '&amp;', '&');
end
