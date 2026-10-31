import json, os, re, sys
# Run from anywhere: paths resolve against the repository root.
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
D=json.load(open("results/derived_transport.json"))
fails=[]; oks=0

def C(model, key): return D[model]["cells"][key]
def chk(label, claimed, actual, tol=0.0006):
    global oks
    if abs(claimed-actual) <= tol: oks += 1
    else: fails.append(f"{label}: paper says {claimed}, data says {actual:.4f}")

# --- Table VII rows: forest msp ---
rows = {
 "Different network": ("network|rf|msp|target_network","network|rf|msp|control",
   (0.960,1.000,0.499,0.451,0.046,1.000,0.488,0.219,0.918,0.043)),
 "Five months later": ("time|rf|msp|later_months","time|rf|msp|control",
   (0.956,0.865,0.639,0.311,0.174,0.925,0.709,0.237,0.914,0.083)),
 "Handshake reduced": ("visibility|rf|msp|visibility_V4","visibility|rf|msp|control",
   (0.954,0.914,0.660,0.290,0.051,1.000,0.740,0.246,0.963,0.029)),
 "Handshake absent":  ("visibility|rf|msp|visibility_V5","visibility|rf|msp|control",
   (0.954,0.914,0.639,0.311,0.030,1.000,0.704,0.226,0.961,0.030)),
 "Unseen unit":       ("instance|rf|msp|unseen_unit","instance|rf|msp|control",
   (0.998,0.882,0.804,0.146,0.154,0.911,0.524,0.317,0.894,0.077)),
}
for name,(sk,ck,v) in rows.items():
    s=C("forest",sk); c=C("forest",ck)
    chk(f"{name} A(c0)", v[0], c["known_acceptance"]["mean"])
    chk(f"{name} R(c0)", v[1], c["unknown_rejection"]["mean"])
    chk(f"{name} A(c)",  v[2], s["known_acceptance"]["mean"])
    chk(f"{name} dA",    v[3], s["delta_A"]["mean"])
    chk(f"{name} dA ci", v[4], s["delta_A"]["ci95"])
    chk(f"{name} R(c)",  v[5], s["unknown_rejection"]["mean"])
    chk(f"{name} R*",    v[6], s["unknown_rejection_at_oracle"]["mean"])
    chk(f"{name} R* ci", v[7], s["unknown_rejection_at_oracle"]["ci95"])
    chk(f"{name} auroc", v[8], s["auroc_unknown"]["mean"])
    chk(f"{name} auroc ci",v[9],s["auroc_unknown"]["ci95"])

# --- Table IX gated proto ---
grows = {
 "g network": ("network|gated|proto|target_network","network|gated|proto|control",(0.260,0.084,0.916,0.459,0.190,0.880)),
 "g time":    ("time|gated|proto|later_months","time|gated|proto|control",(0.183,0.042,0.872,0.583,0.171,0.926)),
 "g V4":      ("visibility|gated|proto|visibility_V4","visibility|gated|proto|control",(0.092,0.065,0.874,0.887,0.063,0.970)),
 "g V5":      ("visibility|gated|proto|visibility_V5","visibility|gated|proto|control",(0.032,0.020,0.874,0.903,0.068,0.975)),
 "g inst":    ("instance|gated|proto|unseen_unit","instance|gated|proto|control",(0.340,0.220,0.893,0.298,0.353,0.711)),
}
for name,(sk,ck,v) in grows.items():
    s=C("gated",sk); c=C("gated",ck)
    chk(f"{name} dA", v[0], s["delta_A"]["mean"]); chk(f"{name} dA ci", v[1], s["delta_A"]["ci95"])
    chk(f"{name} R(c0)", v[2], c["unknown_rejection"]["mean"])
    chk(f"{name} R*", v[3], s["unknown_rejection_at_oracle"]["mean"]); chk(f"{name} R* ci", v[4], s["unknown_rejection_at_oracle"]["ci95"])
    chk(f"{name} auroc", v[5], s["auroc_unknown"]["mean"])

# --- prose claims ---
chk("entropy network dA", 0.461, C("forest","network|rf|entropy|target_network")["delta_A"]["mean"])
chk("entropy network ci", 0.044, C("forest","network|rf|entropy|target_network")["delta_A"]["ci95"])
chk("gated msp V5 dA", -0.004, C("gated","visibility|gated|msp|visibility_V5")["delta_A"]["mean"])
chk("gated msp V5 ci", 0.010, C("gated","visibility|gated|msp|visibility_V5")["delta_A"]["ci95"])
chk("tau used network", 0.003, C("forest","network|rf|msp|target_network")["tau_used"]["mean"], 0.0006)
chk("tau oracle network", 0.569, C("forest","network|rf|msp|target_network")["tau_oracle"]["mean"])
chk("F1 after rej control", 0.929, C("forest","network|rf|msp|control")["known_macro_f1_after_rejection"]["mean"], 0.0006)
chk("F1 after rej target", 0.560, C("forest","network|rf|msp|target_network")["known_macro_f1_after_rejection"]["mean"])
# OSCR range claim 0.805-0.934
osc=[C("forest",k)["oscr"]["mean"] for k in
     ("network|rf|msp|target_network","time|rf|msp|later_months","visibility|rf|msp|visibility_V4",
      "visibility|rf|msp|visibility_V5","instance|rf|msp|unseen_unit")]
chk("OSCR min", 0.805, min(osc)); chk("OSCR max", 0.934, max(osc))
# AUROC range 0.894-0.963
au=[C("forest",k)["auroc_unknown"]["mean"] for k in
    ("network|rf|msp|target_network","time|rf|msp|later_months","visibility|rf|msp|visibility_V4",
     "visibility|rf|msp|visibility_V5","instance|rf|msp|unseen_unit")]
chk("AUROC min", 0.894, min(au)); chk("AUROC max", 0.963, max(au))
# dA range 0.146-0.451
da=[C("forest",k)["delta_A"]["mean"] for k in
    ("network|rf|msp|target_network","time|rf|msp|later_months","visibility|rf|msp|visibility_V4",
     "visibility|rf|msp|visibility_V5","instance|rf|msp|unseen_unit")]
chk("dA min", 0.146, min(da)); chk("dA max", 0.451, max(da))

# --- closed set / cross network / instance / visibility ---
b=json.load(open("results/cf_baseline.json"))["summary"]
chk("gbm", 0.978, b["gbm"]["mean"], 0.0005); chk("rf", 0.976, b["rf"]["mean"], 0.0005)
chk("tcn", 0.964, b["tcn"]["mean"], 0.0005)
chk("n trainable", 27, len(json.load(open("results/cf_baseline.json"))["trainable_classes"]))

for f,(ab,ba) in {"cf_cross_network_rf_FHC_K5.json":(0.144,0.199),
                  "cf_cross_network_rf_FC_K5.json":(0.192,0.262),
                  "cf_cross_network_rf_FC_K3.json":(0.189,0.260)}.items():
    r=json.load(open("results/"+f))["runs"]
    chk(f+" A->B", ab, r["DATA2025v1->DATA2025v2"]["gap"])
    chk(f+" B->A", ba, r["DATA2025v2->DATA2025v1"]["gap"])
    chk(f+" tgt A->B", {0.144:0.856,0.192:0.808,0.189:0.811}[ab], r["DATA2025v1->DATA2025v2"]["target"]["macro_f1"])
    chk(f+" tgt B->A", {0.199:0.801,0.262:0.738,0.260:0.741}[ba], r["DATA2025v2->DATA2025v1"]["target"]["macro_f1"])

ih=json.load(open("results/cf_instance_heldout.json"))["networks"]
chk("inst A f1", 0.720, ih["DATA2025v1"]["unseen_unit"]["macro_f1_over_present_classes"])
chk("inst B f1", 0.775, ih["DATA2025v2"]["unseen_unit"]["macro_f1_over_present_classes"])
chk("inst A ctrl", 1.000, ih["DATA2025v1"]["seen_unit_control"]["macro_f1_over_present_classes"])
chk("inst B ctrl", 0.999, ih["DATA2025v2"]["seen_unit_control"]["macro_f1_over_present_classes"])
chk("inst A windows", 13133, ih["DATA2025v1"]["unseen_unit"]["n_windows"], 0)
chk("inst B windows", 32878, ih["DATA2025v2"]["unseen_unit"]["n_windows"], 0)

v=json.load(open("results/visibility_weekly.json"))
chk("A flows", 3355281, v["DATA2025v1"]["flows"], 0); chk("B flows", 3289497, v["DATA2025v2"]["flows"], 0)
chk("A ch", 107621, v["DATA2025v1"]["client_hellos_found"], 0)
chk("B ch", 118784, v["DATA2025v2"]["client_hellos_found"], 0)
chk("A tls13", 0.763, v["DATA2025v1"]["tls13_share_of_complete"], 0.0006)
chk("B tls13", 0.619, v["DATA2025v2"]["tls13_share_of_complete"], 0.0006)
chk("A hs avail", 0.421, v["DATA2025v1"]["handshake_window_availability"], 0.0006)
chk("B hs avail", 0.316, v["DATA2025v2"]["handshake_window_availability"], 0.0006)
chk("A ech", 668, v["DATA2025v1"]["ech_extension_present_count"], 0)
chk("B ech", 731, v["DATA2025v2"]["ech_extension_present_count"], 0)
chk("A quic", 0.114, v["DATA2025v1"]["quic_window_availability"], 0.0006)
chk("B quic", 0.047, v["DATA2025v2"]["quic_window_availability"], 0.0006)
chk("A sni", 0.994, v["DATA2025v1"]["extension_rates_of_complete"]["server_name (SNI)"], 0.0006)
chk("B sni", 0.931, v["DATA2025v2"]["extension_rates_of_complete"]["server_name (SNI)"], 0.0006)
chk("A alpn", 0.546, v["DATA2025v1"]["extension_rates_of_complete"]["ALPN"], 0.0006)
chk("B alpn", 0.334, v["DATA2025v2"]["extension_rates_of_complete"]["ALPN"], 0.0006)
chk("A devs tls13", 17, v["DATA2025v1"]["devices_offering_tls13"], 0)
chk("B devs tls13", 9, v["DATA2025v2"]["devices_offering_tls13"], 0)
chk("A devs", 35, v["DATA2025v1"]["devices"], 0); chk("B devs", 34, v["DATA2025v2"]["devices"], 0)
chk("A trunc", 0.026, v["DATA2025v1"]["truncation_rate"], 0.0006)
chk("B trunc", 0.012, v["DATA2025v2"]["truncation_rate"], 0.0006)

p=json.load(open("results/probe.json"))
inv={d["dataset"]:d for d in p["inventory"]}
chk("A days", 85, inv["DATA2025v1"]["n_files"], 0); chk("B days", 240, inv["DATA2025v2"]["n_files"], 0)
chk("B retained", 238, inv["DATA2025v2"]["n_after_size_filter"], 0)
chk("A gb", 14.9, inv["DATA2025v1"]["total_gb"], 0.05); chk("B gb", 12.4, inv["DATA2025v2"]["total_gb"], 0.06)
chk("A median", 193, inv["DATA2025v1"]["median_mb"], 0.5); chk("B median", 62, inv["DATA2025v2"]["median_mb"], 0.5)
chk("devices in map", 114, p["n_devices_in_map"], 0)
chk("retained total", 323, inv["DATA2025v1"]["n_after_size_filter"]+inv["DATA2025v2"]["n_after_size_filter"], 0)

print(f"checks passed: {oks}")
if fails:
    print(f"\nMISMATCHES ({len(fails)}):")
    for f in fails: print("  -", f)
    sys.exit(1)
print("all reported values match the measurement records")
