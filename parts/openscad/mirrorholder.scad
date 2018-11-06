/*
    Mirror holder for the head-mounted camera system as described in:

    AF Meyer, J Poort, J O'Keefe, M Sahani, and JF Linden: A head-mounted 
    camera system integrates detailed behavioral monitoring with multichannel 
    electrophysiology in freely moving mice. Neuron, Volume 100, p46-60, 2018.

    author: arne.f.meyer@gmail.com
    licence: GPLv3
*/

// base of the holder width hole for cannula
base_width = 3.5;
base_length = 4.5;
base_height = 2.2;  // suitable for 21 G cannulas

// "arms" for holding the IR mirror tile
arm_height = 1.5;
arm_width = 1;

// mirror dimensions; it might be useful to add 1-2 mm to both dimensions
// and cut off exceeding material using a scapel etc later on. Moreover, cut 
// mirror tiles are typically not perfect squares and for eye and whisker pad
// tracking it is typically useful to have a wide field of view in horizontal 
// rather than in vertical direction.
mirror_width = 7;
mirror_length = 9;

union()
{
    // base
    cube([base_width, base_length, base_height]);

    // left "arm"
    translate([base_width, 0, 0])
        cube([mirror_width, arm_width, arm_height]);

    // upper "arm"
    translate([base_width-arm_width, arm_width, 0])
        cube([arm_width, mirror_length, arm_height]);
}