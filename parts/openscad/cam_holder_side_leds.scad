/*
    Camera holder for the head-mounted camera system as described in:

    AF Meyer, J Poort, J O'Keefe, M Sahani, and JF Linden: A head-mounted 
    camera system integrates detailed behavioral monitoring with multichannel 
    electrophysiology in freely moving mice. Neuron, Volume 100, p46-60, 2018.

    Compatible with Omnivision OV5647-based (e.g., Adafruit 1937) and 
    comparable camera modules. Default dimensions:

    width = 8.5 mm
    height = 8.5 mm
    height_total = 11.8 mm (including electronics on cable)
    thickness = 2 mm

    Note:
    A newer version of the Adafruit 1937 module is available which is a big thicker
    (about 3.1 mm). Change clip_height below accordingly.

    author: arne.f.meyer@gmail.com
    licence: GPLv3
*/

// these are the dimensions of the holder, not the camera
width = 8.7;
height = 12;
thickness = 1.2;
depth = 1;
frame_width = 0.8;

// clips for holding the camera
clip_width = frame_width;
clip_height = 2.5;  // set to 3.1 for newer Adafruit 1937 module
clip_length = 3;
clip_dh = -.5;
clip_dw = .5;

// LED holder (bottom)
led_width = 3;
led_height = 4;

// LED holder (sides)
side_leds = true;
side_led_arm = 5;
side_led_dz = 5;

// block for the cannula that holds the mirror
mirrorblock_width = 3;
mirrorblock_height = 3;
mirrorblock_depth = 4;

// set to mirrorblock=true for making a hole for the cannula in the mirror 
// holder block (only recommended for 3D printers with very fine resolution)
mirrorblock_hole = false;
mirrorblock_hole_dia = 1;

$fn = 100;

union()
{
    // the holder for the camera (with frame)
    difference()
    {
        cube([width+2*frame_width, height+frame_width, thickness+depth], center=false);
        translate([frame_width, frame_width, thickness])
            cube([width, height+.1, depth+.1], center=false);
    }

    // clips for the camera
    translate([0, .5*height, thickness+depth])
        cube([clip_width, clip_length, clip_height-depth]);
    translate([0, .5*height, thickness+clip_height])
        cube([clip_width+clip_dw, clip_length, clip_height-depth+clip_dh]);

    translate([width+frame_width, .5*height, thickness+depth])
        cube([clip_width, clip_length, clip_height-depth]);
    translate([width+frame_width-clip_dw, .5*height, thickness+clip_height])
        cube([clip_width+clip_dw, clip_length, clip_height-depth+clip_dh]);

    // LED holder
    translate([.5*frame_width+.5*width-.5*led_width, 0, frame_width])
        cube([led_width, frame_width, thickness+led_height]);

    if (side_leds)
    {
        h = thickness+depth;
        translate([-side_led_arm, 0, 0])
        {
            cube([side_led_arm, frame_width, h]);
            translate([0, 0, h])
                cube([h, frame_width, side_led_dz]);
        }

        translate([width+2*frame_width, 0, 0])
        {
            cube([side_led_arm, frame_width, h]);
            translate([side_led_arm-h, 0, h])
                cube([h, frame_width, side_led_dz]);
        }
    }

    // block for the cannular holding the mirror
    translate([-mirrorblock_width, height - mirrorblock_height+depth, 0])
    {
        difference()
        {
            cube([mirrorblock_width, mirrorblock_height, mirrorblock_depth]);

            if (mirrorblock_hole)
            {
                translate([.5*mirrorblock_width, .5*mirrorblock_height, -.1])
                    cylinder(d=mirrorblock_hole_dia, h=mirrorblock_height+height+.2);
            }
        }
    }
}
