echo "### follow-up batch"
python scripts/poe_compare.py runs/rw_poe10c05_g_ad4_r4 runs/rw_poe10c05_g_ad4_r4_f03 runs/rw_poe5c05_g_r4 runs/rw_poe5c1_g_r4 runs/rw_poe10c05_g_ad4_r4_hi runs/rw_poe10c05_g_bang4_r4_dec
echo "### summary"
python scripts/summary_table.py --bank decoy_rw runs/rw_poe5c1_g_r4 runs/rw_poe10c05_g_ad4_r4_f03 runs/rw_poe10c05_g_ad4_r4_hi runs/rw128_poe10c05_g_bang4_r4_dec runs/rw256_poe10c05_g_bang4_r4_dec runs/rw_poe10c05_g_bang4_r4_dec
echo "### attacks: decoy push beta 10"
python scripts/attack_table.py runs/attacks_rw_poe10c05_g_bang4_r4_dec runs/rw_poe10c05_g_bang4_r4_dec
echo "### attacks: maintenance push (f03)"
python scripts/attack_table.py runs/attacks_rw_poe10c05_g_ad4_r4_f03 runs/rw_poe10c05_g_ad4_r4_f03
echo "### attacks: twin without maintenance push"
python scripts/attack_table.py runs/attacks_rw_poe10c05_g_ad4_r4 runs/rw_poe10c05_g_ad4_r4
