classdef MaetCache < handle
%MAETCACHE  The joint fields of a lazy density, built once and shared.
%
%   buildMaet gives every lazy (skinny) density a MaetCache in its field
%   lazyCache. A density is a struct, passed by value, so a consumer that
%   builds the density's joint (expensive) fields cannot leave them on the
%   caller's struct, and without a cache each call would build them again.
%   A handle is shared by every copy of the struct instead: the first call
%   that needs the fields (INTERNAL.ENSUREMAETEXPENSIVE) stores the full
%   density here, and later calls, on the struct or any copy of it, take
%   it from here. Python keeps the fields on the density object, built
%   once, to the same effect.
%
%   Properties:
%     filled  true once a build is stored;
%     inputs  the fields of the skinny density that the stored build was
%             made from. The cache is used only while they are unchanged,
%             so a copy whose fields have been edited is rebuilt rather
%             than given stale fields, and the rebuild replaces what the
%             cache holds;
%     dens    the full density that build returned.
%
%   The properties are transient: a density saved to a MAT-file is saved
%   without them, and builds its joint fields again on first use after
%   loading. A parfor worker likewise receives an empty cache and fills
%   its own.
%
%   See also INTERNAL.ENSUREMAETEXPENSIVE, BUILDMAET.

    properties (Transient)
        filled = false
        inputs = []
        dens = []
    end
end
