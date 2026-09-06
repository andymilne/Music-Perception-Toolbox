"""Generate the small score fixtures shared by the Python and MATLAB tests."""
import struct, os, sys, zipfile
out = sys.argv[1]
def varlen(v):
    o=[v&0x7f]; v>>=7
    while v: o.append((v&0x7f)|0x80); v>>=7
    return bytes(reversed(o))
def track(events, eot_delta=0):
    data=b"".join(varlen(d)+e for d,e in events)+varlen(eot_delta)+b"\xff\x2f\x00"
    return b"MTrk"+struct.pack(">I",len(data))+data
tpq=480
t0=[(0,b"\xff\x51\x03"+(500000).to_bytes(3,"big")),(0,b"\xff\x58\x04\x03\x02\x18\x08"),(0,b"\xff\x03\x05Tempo"),(4*tpq,b"\xff\x51\x03"+(1000000).to_bytes(3,"big"))]
t1=[(0,b"\xff\x03\x06Melody"),(0,b"\x90\x3c\x60"),(tpq,b"\x80\x3c\x00"),(0,b"\x90\x40\x50"),(tpq//2,b"\x90\x40\x00"),(tpq//2,b"\x90\x43\x70"),(2*tpq,b"\x80\x43\x00"),(0,b"\x90\x48\x40"),(tpq,b"\x80\x48\x00")]
t2=[(0,b"\xff\x03\x06Chords"),(0,b"\x91\x30\x64"),(0,b"\x34\x64"),(0,b"\x37\x64"),(2*tpq,b"\x81\x30\x40"),(0,b"\x34\x40"),(0,b"\x37\x40"),(0,b"\x91\x35\x64"),(0,b"\x39\x64")]
hdr=b"MThd"+struct.pack(">IHHH",6,1,3,tpq)
os.makedirs(out,exist_ok=True)
open(os.path.join(out,"score_small.mid"),"wb").write(hdr+track(t0)+track(t1)+track(t2, eot_delta=2*tpq))
xml=open(os.path.join(os.path.dirname(os.path.abspath(__file__)),"score_small_template.musicxml")).read()
open(os.path.join(out,"score_small.musicxml"),"w").write(xml)
with zipfile.ZipFile(os.path.join(out,"score_small.mxl"),"w") as zf:
    zf.writestr("META-INF/container.xml",'<?xml version="1.0"?><container><rootfiles><rootfile full-path="score_small.musicxml"/></rootfiles></container>')
    zf.writestr("score_small.musicxml",xml)
