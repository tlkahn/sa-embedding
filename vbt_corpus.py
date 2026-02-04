# VBT verses in Devanagari (168 verses from Rails database)
VBT_CORPUS = [
    "श्रुतं देव मया सर्वं रुद्रयामलसम्भवम् / त्रिकभेदम् अशेषेण सारात् सारविभागशः",  # 0
    "अद्यापि न निवृत्तो मे संशयः परमेश्वर / किंरूपं तत्त्वतो देव शब्दराशिकलामयम्",  # 1
    "किं वा नवात्मभेदेन भैरवे भैरवाकृतौ / त्रिशिरोभेदभिन्नं वा किं वा शक्तित्रयात्मकम्",  # 2
    "नादबिन्दुमयं वापि किं चन्द्रार्धनिरोधिकाः / चक्रारूढम् अनच्कं वा किं वा शक्तिस्वरूपकम्",  # 3
    "परापरायाः सकलम् अपरायाश् च वा पुनः / पराया यदि तद्वत् स्यात् परत्वं तद् विरुध्यते",  # 4
    "न हि वर्णविभेदेन देहभेदेन वा भवेत् / परत्वं निष्कलत्वेन सकलत्वे न तद् भवेत्",  # 5
    "प्रसादं कुरु मे नाथ निःशेषं छिन्द्धि संशयम् ।",  # 6
    "भैरव उवाच ।",  # 7
    "साधु साधु त्वया पृष्टं तन्त्रसारम् इदम् प्रिये",  # 8
    "गूहनीयतमम् भद्रे तथापि कथयामि ते / यत्किंचित् सकलं रूपं भैरवस्य प्रकीर्तितम्",  # 9
    "तद् असारतया देवि विज्ञेयं शक्रजालवत् / मायास्वप्नोपमं चैव गन्धर्वनगरभ्रमम्",  # 10
    "ध्यानार्थम् भ्रान्तबुद्धीनां क्रियाडम्बरवर्तिनाम् / केवलं वर्णितम् पुंसां विकल्पनिहतात्मनाम्",  # 11
    "तत्त्वतो न नवात्मासौ शब्दराशिर् न भैरवः / न चासौ त्रिशिरा देवो न च शक्तित्रयात्मकः",  # 12
    "नादबिन्दुमयो वापि न चन्द्रार्धनिरोधिकाः / न चक्रक्रमसम्भिन्नो न च शक्तिस्वरूपकः",  # 13
    "अप्रबुद्धमतीनां हि एता बलविभीषिकाः / मातृमोदकवत् सर्वं प्रवृत्त्यर्थं उदाहृतम्",  # 14
    "दिक्कालकलनोन्मुक्ता देशोद्देशाविशेषिनी / व्यपदेष्टुम् अशक्यासाव् अकथ्या परमार्थतः",  # 15
    "अन्तःस्वानुभवानन्दा विकल्पोन्मुक्तगोचरा / यावस्था भरिताकारा भैरवी भैरवात्मनः",  # 16
    "तद् वपुस् तत्त्वतो ज्ञेयं विमलं विश्वपूरणम् / एवंविधे परे तत्त्वे कः पूज्यः कश् च तृप्यति",  # 17
    "एवंविधा भैरवस्य यावस्था परिगीयते / सा परा पररूपेण परा देवी प्रकीर्तिता",  # 18
    "शक्तिशक्तिमतोर् यद्वद् अभेदः सर्वदा स्थितः / अतस् तद्धर्मधर्मित्वात् परा शक्तिः परात्मनः",  # 19
    "न वह्नेर् दाहिका शक्तिर् व्यतिरिक्ता विभाव्यते / केवलं ज्ञानसत्तायाम् प्रारम्भो ऽयम् प्रवेशने",  # 20
    "शक्त्यवस्थाप्रविष्टस्य निर्विभागेन भावना / तदासौ शिवरूपी स्यात् शैवी मुखम् इहोच्यते",  # 21
    "यथालोकेन दीपस्य किरणैर् भास्करस्य च / ज्ञायते दिग्विभागादि तद्वच् छक्त्या शिवः प्रिये",  # 22
    "देवदेव त्रिशूलाङ्क कपालकृतभूषण / दिग्देशकालशून्या च व्यपदेशविवर्जिता",  # 23
    "यावस्था भरिताकारा भैरवस्योपलभ्यते / कैर् उपायैर् मुखं तस्य परा देवि कथम् भवेत् / यथा सम्यग् अहं वेद्मि तथा मे ब्रूहि भैरव",  # 24
    "ऊर्ध्वे प्राणो ह्य् अधो जीवो विसर्गात्मा परोच्चरेत् / उत्पत्तिद्वितयस्थाने भरणाद् भरिता स्थितिः",  # 25
    "मरुतो ऽन्तर् बहिर् वापि वियद्युग्मानिवर्तनात् / भैरव्या भैरवस्येत्थम् भैरवि व्यज्यते वपुः",  # 26
    "न व्रजेन् न विशेच् छक्तिर् मरुद्रूपा विकासिते / निर्विकल्पतया मध्ये तया भैरवरूपता",  # 27
    "कुम्भिता रेचिता वापि पूरिता वा यदा भवेत् / तदन्ते शान्तनामासौ शक्त्या शान्तः प्रकाशते",  # 28
    "आमूलात् किरणाभासां सूक्ष्मात् सूक्ष्मतरात्मिकम् / चिन्तयेत् तां द्विषट्कान्ते श्याम्यन्तीम् भैरवोदयः",  # 29
    "उद्गच्छन्तीं तडित्रूपाम् प्रतिचक्रं क्रमात् क्रमम् / ऊर्ध्वं मुष्टित्रयं यावत् तावद् अन्ते महोदयः",  # 30
    "क्रमद्वादशकं सम्यग् द्वादशाक्षरभेदितम् / स्थूलसूक्ष्मपरस्थित्या मुक्त्वा मुक्त्वान्ततः शिवः",  # 31
    "तयापूर्याशु मूर्धान्तं भङ्क्त्वा भ्रूक्षेपसेतुना / निर्विकल्पं मनः कृत्वा सर्वोर्ध्वे सर्वगोद्गमः",  # 32
    "शिखिपक्षैश् चित्ररूपैर् म।ङ्दलैः शून्यपञ्चकम् / ध्यायतो ऽनुत्तरे शून्ये प्रवेशो हृदये भवेत्",  # 33
    "ईदृशेन क्रमेणैव यत्र कुत्रापि चिन्तना / शून्ये कुड्ये परे पात्रे स्वयं लीना वरप्रदा",  # 34
    "कपालान्तर् मनो न्यस्य तिष्ठन् मीलितलोचनः / क्रमेण मनसो दार्ढ्यात् लक्षयेत् लष्यम् उत्तमम्",  # 35
    "मध्यनाडी मध्यसंस्था बिससूत्राभरूपया / ध्यातान्तर्व्योमया देव्या तया देवः प्रकाशते",  # 36
    "कररुद्धदृगस्त्रेण भ्रूभेदाद् द्वाररोधनात् / दृष्टे बिन्दौ क्रमाल् लीने तन्मध्ये परमा स्थितिः",  # 37
    "धामान्तःक्षोभसम्भूतसूक्ष्माग्नितिलकाकृतिम् / बिन्दुं शिखान्ते हृदये लयान्ते ध्यायतो लयः",  # 38
    "अनाहते पात्रकर्णे ऽभग्नशब्दे सरिद्द्रुते / शब्दब्रह्मणि निष्णातः परम् ब्रह्माधिगच्छति",  # 39
    "प्रणवादिसमुच्चारात् प्लुतान्ते शून्यभावानात् / शून्यया परया शक्त्या शून्यताम् एति भैरवि",  # 40
    "यस्य कस्यापि वर्णस्य पूर्वान्ताव् अनुभावयेत् / शून्यया शून्यभूतो ऽसौ शून्याकारः पुमान् भवेत्",  # 41
    "तन्त्र्यादिवाद्यशब्देषु दीर्घेषु क्रमसंस्थितेः / अनन्यचेताः प्रत्यन्ते परव्योमवपुर् भवेत्",  # 42
    "पिण्डमन्त्रस्य सर्वस्य स्थूलवर्णक्रमेण तु / अर्धेन्दुबिन्दुनादान्तः शून्योच्चाराद् भवेच् छिवः",  # 43
    "निजदेहे सर्वदिक्कं युगपद् भावयेद् वियत् / निर्विकल्पमनास् तस्य वियत् सर्वम् प्रवर्तते",  # 44
    "पृष्टशून्यं मूलशून्यं युगपद् भावयेच् च यः / शरीरनिरपेक्षिण्या शक्त्या शून्यमना भवेत्",  # 45
    "पृष्टशून्यं मूलशून्यं हृच्छून्यम् भावयेत् स्थिरम् / युगपन् निर्विकल्पत्वान् निर्विकल्पोदयस् ततः",  # 46
    "तनूदेशे शून्यतैव क्षणमात्रं विभावयेत् / निर्विकल्पं निर्विकल्पो निर्विकल्पस्वरूपभाक्",  # 47
    "सर्वं देहगतं द्रव्यं वियद्व्याप्तं मृगेक्षणे / विभावयेत् ततस् तस्य भावना सा स्थिरा भवेत्",  # 48
    "देहान्तरे त्वग्विभागम् भित्तिभूतं विचिन्तयेत् / न किंचिद् अन्तरे तस्य ध्यायन्न् अध्येयभाग् भवेत्",  # 49
    "हृद्याकाशे निलीनाक्षः पद्मसम्पुटमध्यगः / अनन्यचेताः सुभगे परं सौभाग्यम् आप्नुयात्",  # 50
    "सर्वतः स्वशरीरस्य द्वादशान्ते मनोलयात् / दृढबुद्धेर् दृढीभूतं तत्त्वलक्ष्यम् प्रवर्तते",  # 51
    "यथा तथा यत्र तत्र द्वादशान्ते मनः क्षिपेत् / प्रतिक्षणं क्षीणवृत्तेर् वैलक्षण्यं दिनैर् भवेत्",  # 52
    "कालाग्निना कालपदाद् उत्थितेन स्वकम् पुरम् / प्लुष्टम् विचिन्तयेद् अन्ते शान्ताभासस् तदा भवेत्",  # 53
    "एवम् एव जगत् सर्वं दग्धं ध्यात्वा विकल्पतः / अनन्यचेतसः पुंसः पुम्भावः परमो भवेत्",  # 54
    "स्वदेहे जगतो वापि सूक्ष्मसूक्ष्मतराणि च / तत्त्वानि यानि निलयं ध्यात्वान्ते व्यज्यते परा",  # 55
    "पिनां च दुर्बलां शक्तिं ध्यात्वा द्वादशगोचरे / प्रविश्य हृदये ध्यायन् मुक्तः स्वातन्त्र्यम् आप्नुयात्",  # 56
    "भुवनाध्वादिरूपेण चिन्तयेत् क्रमशो ऽखिलम् / स्थूलसूक्ष्मपरस्थित्या यावद् अन्ते मनोलयः",  # 57
    "अस्य सर्वस्य विश्वस्य पर्यन्तेषु समन्ततः / अध्वप्रक्रियया तत्त्वं शैवं ध्यत्वा महोदयः",  # 58
    "विश्वम् एतन् महादेवि शून्यभूतं विचिन्तयेत् / तत्रैव च मनो लीनं ततस् तल्लयभाजनम्",  # 59
    "घतादिभाजने दृष्टिम् भित्तिस् त्यक्त्वा विनिक्षिपेत् / तल्लयं तत्क्षणाद् गत्वा तल्लयात् तन्मयो भवेत्",  # 60
    "निर्वृक्षगिरिभित्त्यादिदेशे दृष्टिं विनिक्षिपेत् / विलीने मानसे भावे वृत्तिक्षिणः प्रजायते",  # 61
    "उभयोर् भावयोर् ज्ञाने ध्यात्वा मध्यं समाश्रयेत् / युगपच् च द्वयं त्यक्त्वा मध्ये तत्त्वम् प्रकाशते",  # 62
    "भावे त्यक्ते निरुद्धा चिन् नैव भावान्तरं व्रजेत् / तदा तन्मध्यभावेन विकसत्यति भावना",  # 63
    "सर्वं देहं चिन्मयं हि जगद् वा परिभावयेत् / युगपन् निर्विकल्पेन मनसा परमोदयः",  # 64
    "वायुद्वयस्य संघट्टाद् अन्तर् वा बहिर् अन्ततः / योगी समत्वविज्ञानसमुद्गमनभाजनम्",  # 65
    "सर्वं जगत् स्वदेहं वा स्वानन्दभरितं स्मरेत् / युगपत् स्वामृतेनैव परानन्दमयो भवेत्",  # 66
    "कुहनेन प्रयोगेण सद्य एव मृगेक्षणे / समुदेति महानन्दो येन तत्त्वं प्रकाशते",  # 67
    "सर्वस्रोतोनिबन्धेन प्राणशक्त्योर्ध्वया शनैः / पिपीलस्पर्शवेलायाम् प्रथते परमं सुखम्",  # 68
    "वह्नेर् विषस्य मध्ये तु चित्तं सुखमयं क्षिपेत् / केवलं वायुपूर्णं वा स्मरानन्देन युज्यते",  # 69
    "शक्तिसंगमसंक्षुब्धशक्त्यावेशावसानिकम् / यत् सुखम् ब्रह्मतत्त्वस्य तत् सुखं स्वाक्यम् उच्यते",  # 70
    "लेहनामन्थनाकोटैः स्त्रीसुखस्य भरात् स्मृतेः / शक्त्यभावे ऽपि देवेशि भवेद् आनन्दसम्प्लवः",  # 71
    "आनन्दे महति प्राप्ते दृष्टे वा बान्धवे चिरात् / आनन्दम् उद्गतं ध्यात्वा तल्लयस् तन्मना भवेत्",  # 72
    "जग्धिपानकृतोल्लासरसानन्दविजृम्भणात् / भावयेद् भरितावस्थां महानन्दस् ततो भवेत्",  # 73
    "गितादिविषयास्वादासमसौख्यैकतात्मनः / योगिनस् तन्मयत्वेन मनोरूढेस् तदात्मता",  # 74
    "यत्र यत्र मनस् तुष्टिर् मनस् तत्रैव धारयेत् / तत्र तत्र परानन्दस्वारूपं सम्प्रवर्तते",  # 75
    "अनागतायां निद्रायाम् प्रणष्टे बाह्यगोचरे / सावस्था मनसा गम्या परा देवी प्रकाशते",  # 76
    "तेजसा सूर्यदीपादेर् आकाशे शबलीकृते / दृष्टिर् निवेश्या तत्रैव स्वात्मरूपम् प्रकाशते",  # 77
    "करङ्किण्या क्रोधनया भैरव्या लेलिहानया / खेचर्या दृष्टिकाले च परावाप्तिः प्रकाशते",  # 78
    "मृद्वासने स्फिजैकेन हस्तपादौ निराश्रयम् / निधाय तत्प्रसङ्गेन परा पूर्णा मतिर् भवेत्",  # 79
    "उपविश्यासने सम्यग् बाहू कृत्वार्धकुञ्चितौ / कक्षव्योम्नि मनः कुर्वन् शमम् आयाति तल्लयात्",  # 80
    "स्थूलरूपस्य भावस्य स्तब्धां दृष्टिं निपात्य च / अचिरेण निराधारं मनः कृत्वा शिवं व्रजेत्",  # 81
    "मध्यजिह्वे स्फारितास्ये मध्ये निक्षिप्य चेतनाम् / होच्चारं मनसा कुर्वंस् ततः शान्ते प्रलीयते",  # 82
    "आसने शयने स्थित्वा निराधारं विभावयन् / स्वदेहं मनसि क्षिणे क्षणात् क्षीणाशयो भवेत्",  # 83
    "चलासने स्थितस्याथ शनैर् वा देहचालनात् / प्रशान्ते मानसे भावे देवि दिव्यौघम् आप्नुयात्",  # 84
    "आकाशं विमलम् पश्यन् कृत्वा दृष्टिं निरन्तराम् / स्तब्धात्मा तत्क्षणाद् देवि भैरवं वपुर् आप्नुयात्",  # 85
    "लीनं मूर्ध्नि वियत् सर्वम् भैरवत्वेन भावयेत् / तत् सर्वम् भैरवाकारतेजस्तत्त्वं समाविशेत्",  # 86
    "किञ्चिज् ज्ञातं द्वैतदायि बाह्यालोकस् तमः पुनः / विश्वादि भैरवं रूपं ज्ञात्वानन्तप्रकाशभृत्",  # 87
    "एवम् एव दुर्निशायां कृष्णपक्षागमे चिरम् / तैमिरम् भावयन् रूपम् भैरवं रूपम् एष्यति",  # 88
    "एवम् एव निमील्यादौ नेत्रे कृष्णाभम् अग्रतः / प्रसार्य भैरवं रूपम् भावयंस् तन्मयो भवेत्",  # 89
    "यस्य कस्येन्द्रियस्यापि व्याघाताच् च निरोधतः / प्रविष्टस्याद्वये शून्ये तत्रैवात्मा प्रकाशते",  # 90
    "अबिन्दुम् अविसर्गं च अकारं जपतो महान् / उदेति देवि सहसा ज्ञानौघः परमेश्वरः",  # 91
    "वर्णस्य सविसर्गस्य विसर्गान्तं चितिं कुरु / निराधारेण चित्तेन स्पृशेद् ब्रह्म सनातनम्",  # 92
    "व्योमाकारं स्वम् आत्मानं ध्यायेद् दिग्भिर् अनावृतम् / निराश्रया चितिः शक्तिः स्वरूपं दर्शयेत् तदा",  # 93
    "किंचिद् अङ्गं विभिद्यादौ तीक्ष्णसूच्यादिना ततः / तत्रैव चेतनां युक्त्वा भैरवे निर्मला गतिः",  # 94
    "चित्ताद्यन्तःकृतिर् नास्ति ममान्तर् भावयेद् इति / विकल्पानाम् अभावेन विकल्पैर् उज्झितो भवेत्",  # 95
    "माया विमोहिनी नाम कलायाः कलनं स्थितम् / इत्यादिधर्मं तत्त्वानां कलयन् न पृथग् भवेत्",  # 96
    "झगितीच्छां समुत्पन्नाम् अवलोक्य शमं नयेत् / यत एव समुद्भूता ततस् तत्रैव लीयते",  # 97
    "यदा ममेच्छा नोत्पन्ना ज्ञानं वा कस् तदास्मि वै / तत्त्वतो =ब्९हं तथाभूतस् तल्लीनस् तन्मना भवेत्",  # 98
    "इच्छायाम् अथवा ज्ञाने जाते चित्तं निवेशयेत् / आत्मबुद्ध्यानन्यचेतास् ततस् तत्त्वार्थदर्शनम्",  # 99
    "निर्निमित्तम् भवेज् ज्ञानं निराधारम् भ्रमात्मकम् / तत्त्वतः कस्यचिन् नैतद् एवम्भावी शिवः प्रिये",  # 100
    "चिद्धर्मा सर्वदेहेषु विशेषो नास्ति कुत्रचित् / अतश् च तन्मयं सर्वम् भावयन् भवजिज् जनः",  # 101
    "कामक्रोधलोभमोहमदमात्सर्यगोचरे / बुद्धिं निस्तिमितां कृत्वा तत् तत्त्वम् अवशिष्यते",  # 102
    "इन्द्रजालमयं विश्वं व्यस्तं वा चित्रकर्मवत् / भ्रमद् वा ध्यायतः सर्वम् पश्यतश् च सुखोद्गमः",  # 103
    "न चित्तं निक्षिपेद् दुःखे न सुखे वा परिक्षिपेत् / भैरवि ज्ञायतां मध्ये किं तत्त्वम् अवशिष्यते",  # 104
    "विहाय निजदेहस्थं सर्वत्रास्मीति भावयन् / दृढेन मनसा दृष्ट्या नान्येक्षिण्या सुखी भवेत्",  # 105
    "घटादौ यच् च विज्ञानम् इच्छाद्यं वा ममान्तरे / नैव सर्वगतं जातम् भावयन् इति सर्वगः",  # 106
    "ग्राह्यग्राहकसंवित्तिः सामान्या सर्वदेहिनाम् / योगिनां तु विशेषो =ब्९स्ति सम्बन्धे सावधानता",  # 107
    "स्ववद् अन्यशरीरे =ब्९पि संवित्तिम् अनुभावयेत् / अपेक्षां स्वशरीरस्य त्यक्त्वा व्यापी दिनैर् भवेत्",  # 108
    "निराधारं मनः कृत्वा विकल्पान् न विकल्पयेत् / तदात्मपरमात्मत्वे भैरवो मृगलोचने",  # 109
    "सर्वज्ञः सर्वकर्ता च व्यापकः परमेश्वरः / स एवाहं शैवधर्मा इति दार्ढ्याच् छिवो भवेत्",  # 110
    "जलस्येवोर्मयो वह्नेर् ज्वालाभङ्ग्यः प्रभा रवेः / ममैव भैरवस्यैता विश्वभङ्ग्यो विभेदिताः",  # 111
    "भ्रान्त्वा भ्रान्त्वा शरीरेण त्वरितम् भुवि पातनात् / क्षोभशक्तिविरामेण परा संजायते दशा",  # 112
    "आधारेष्व् अथवा ऽशक्त्या ऽज्ञानाच् चित्तलयेन वा / जातशक्तिसमावेशक्षोभान्ते भैरवं वपुः",  # 113
    "सम्प्रदायम् इमम् देवि शृणु सम्यग् वदाम्य् अहम् / कैवल्यं जायते सद्यो नेत्रयोः स्तब्धमात्रयोः",  # 114
    "संकोचं कर्णयोः कृत्वा ह्य् अधोद्वारे तथैव च / अनच्कम् अहलं ध्यायन् विशेद् ब्रह्म सनातनम्",  # 115
    "कूपादिके महागर्ते स्थित्वोपरि निरीक्षणात् / अविकल्पमतेः सम्यक् सद्यस् चित्तलयः स्फुटम्",  # 116
    "यत्र यत्र मनो याति बाह्ये वाभ्यन्तरे ऽपि वा / तत्र तत्र शिवावास्था व्यापकत्वात् क्व यास्यति",  # 117
    "यत्र यत्राक्षमार्गेण चैतन्यं व्यज्यते विभोः / तस्य तन्मात्रधर्मित्वाच् चिल्लयाद् भरितात्मता",  # 118
    "क्षुताद्यन्ते भये शोके गह्वरे वा रणाद् द्रुते / कुतूहलेक्षुधाद्यन्ते ब्रह्मसत्तामयी दशा",  # 119
    "वस्तुषु स्मर्यमाणेषु दृष्टे देशे मनस् त्यजेत् / स्वशरीरं निराधारं कृत्वा प्रसरति प्रभुः",  # 120
    "क्वचिद् वस्तुनि विन्यस्य शनैर् दृष्टिं निवर्तयेत् / तज् ज्ञानं चित्तसहितं देवि शून्यालायो भवेत्",  # 121
    "भक्त्युद्रेकाद् विरक्तस्य यादृशी जायते मतिः / सा शक्तिः शाङ्करी नित्यम् भवयेत् तां ततः शिवः",  # 122
    "वस्त्वन्तरे वेद्यमाने सर्ववस्तुषु शून्यता / ताम् एव मनसा ध्यात्वा विदितो ऽपि प्रशाम्यति",  # 123
    "किंचिज्ज्ञैर् या स्मृता शुद्धिः सा शुद्धिः शम्भुदर्शने / न शुचिर् ह्य् अशुचिस् तस्मान् निर्विकल्पः सुखी भवेत्",  # 124
    "सर्वत्र भैरवो भावः सामान्येष्व् अपि गोचरः / न च तद्व्यतिरेक्तेण परो ऽस्तीत्य् अद्वया गतिः",  # 125
    "समः शत्रौ च मित्रे च समो मानावमानयोः ।/ ब्रह्मणः परिपूर्णत्वात् इति ज्ञात्वा सुखी भवेत्",  # 126
    "न द्वेषम् भावयेत् क्वापि न रागम् भावयेत् क्वचित् / रागद्वेषविनिर्मुक्तौ मध्ये ब्रह्म प्रसर्पति",  # 127
    "यद् अवेद्यं यद् अग्राह्यं यच् छून्यं यद् अभावगम् / तत् सर्वम् भैरवम् भाव्यं तदन्ते बोधसम्भवः",  # 128
    "नित्ये निराश्रये शून्ये व्यापके कलनोज्झिते / बाह्याकाशे मनः कृत्वा निराकाशं समाविशेत्",  # 129
    "यत्र यत्र मनो याति तत् तत् तेनैव तत्क्षणम् / परित्यज्यानवस्थित्या निस्तरङ्गस् ततो भवेत्",  # 130
    "भया सर्वं रवयति सर्वदो व्यापको ऽखिले / इति भैरवशब्दस्य सन्ततोच्चारणाच् छिवः",  # 131
    "अहं ममेदम् इत्यादि प्रतिपत्तिप्रसङ्गतः / निराधारे मनो याति तद्ध्यानप्रेरणाच् छमी",  # 132
    "नित्यो विभुर् निराधारो व्यापकश् चाखिलाधिपः / शब्दान् प्रतिक्षणं ध्यायन् कृतार्थो ऽर्थानुरूपतः",  # 133
    "अतत्त्वम् इन्द्रजालाभम् इदं सर्वम् अवस्थितम् / किं तत्त्वम् इन्द्रजालस्य इति दार्ढ्याच् छमं व्रजेत्",  # 134
    "आत्मनो निर्विकारस्य क्व ज्ञानं क्व च वा क्रिया / ज्ञानायत्ता बहिर्भावा अतः शून्यम् इदं जगत्",  # 135
    "न मे बन्धो न मोक्षो मे भीतस्यैता विभीषिकाः / प्रतिबिम्बम् इदम् बुद्धेर् जलेष्व् इव विवस्वतः",  # 136
    "इन्द्रियद्वारकं सर्वं सुखदुःखादिसंगमम् / इतीन्द्रियाणि संत्यज्य स्वस्थः स्वात्मनि वर्तते",  # 137
    "ज्ञानप्रकाशकं सर्वं सर्वेणात्मा प्रकाशकः / एकम् एकस्वभावत्वात् ज्ञानं ज्ञेयं विभाव्यते",  # 138
    "मानसं चेतना शक्तिर् आत्मा चेति चतुष्टयम् / यदा प्रिये परिक्षीणं तदा तद् भैरवं वपुः",  # 139
    "निस्तरङ्गोपदेशानां शतम् उक्तं समासतः / द्वादशाभ्यधिकं देवि यज् ज्ञात्वा ज्ञानविज् जनः",  # 140
    "अत्र चैकतमे युक्तो जायते भैरवः स्वयम् / वाचा करोति कर्माणि शापानुग्रहकारकः",  # 141
    "अजरामरताम् एति सो ऽणिमादिगुणान्वितः / योगिनीनाम् प्रियो देवि सर्वमेलापकाधिपः",  # 142
    "जीवन्न् अपि विमुक्तो ऽसौ कुर्वन्न् अपि न लिप्यते ।",  # 143
    "श्री देवी उवाच ।",  # 144
    "इदं यदि वपुर् देव परायाश् च महेश्वर",  # 145
    "एवमुक्तव्यवस्थायां जप्यते को जपश् च कः / ध्यायते को महानाथ पूज्यते कश् च तृप्यति",  # 146
    "हूयते कस्य वा होमो यागः कस्य च किं कथम् ।",  # 147
    "श्री भैरव उवाच ।",  # 148
    "एषात्र प्रक्रिया बाह्या स्थूलेष्व् एव मृगेक्षणे",  # 149
    "भूयो भूयः परे भावे भावना भाव्यते हि या / जपः सो ऽत्र स्वयं नादो मन्त्रात्मा जप्य ईदृशः",  # 150
    "ध्यानं हि निश्चला बुद्धिर् निराकारा निराश्रया / न तु ध्यानं शरीराक्षिमुखहस्तादिकल्पना",  # 151
    "पूजा नाम न पुष्पाद्यैर् या मतिः क्रियते दृढा / निर्विकल्पे महाव्योम्नि सा पूजा ह्य् आदराल् लयः",  # 152
    "अत्रैकतमयुक्तिस्थे योत्पद्येत दिनाद् दिनम् / भरिताकारता सात्र तृप्तिर् अत्यन्तपूर्णता",  # 153
    "महाशून्यालये वह्नौ भूताक्षविषयादिकम् / हूयते मनसा सार्धं स होमश् चेतनास्रुचा",  # 154
    "यागो ऽत्र परमेशानि तुष्टिर् आनन्दलक्षणा / क्षपणात् सर्वपापानां त्राणात् सर्वस्य पार्वति",  # 155
    "रुद्रशक्तिसमावेशस् तत् क्षेत्रम् भावना परा / अन्यथा तस्य तत्त्वस्य का पूजा काश् च तृप्यति",  # 156
    "स्वतन्त्रानन्दचिन्मात्रसारः स्वात्मा हि सर्वतः / आवेशनं तत्स्वरूपे स्वात्मनः स्नानम् ईरितम्",  # 157
    "यैर् एव पूज्यते द्रव्यैस् तर्प्यते वा परापरः / यश् चैव पूजकः सर्वः स एवैकः क्व पूजनम्",  # 158
    "व्रजेत् प्राणो विशेज् जीव इच्छया कुटिलाकृतिः / दीर्घात्मा सा महादेवी परक्षेत्रम् परापरा",  # 159
    "अस्याम् अनुचरन् तिष्ठन् महानन्दमये ऽध्वरे / तया देव्या समाविष्टः परम् भैरवम् आप्नुयात्",  # 160
    "षट्शतानि दिवा रात्रौ सहस्राण्येकविंशतिः / जपो देव्याः समुद्दिष्टः सुलभो दुर्लभो जडैः",  # 161
    "इत्य् एतत् कथितं देवि परमामृतम् उत्तमम् / एतच् च नैव कस्यापि प्रकाश्यं तु कदाचन",  # 162
    "परशिष्ये खले क्रूरे अभक्ते गुरुपादयोः / निर्विकल्पमतीनां तु वीराणाम् उन्नतात्मनाम्",  # 163
    "भक्तानां गुरुवर्गस्य दातव्यं निर्विशङ्कया / ग्रामो राज्यम् पुरं देशः पुत्रदारकुटुम्बकम्",  # 164
    "सर्वम् एतत् परित्यज्य ग्राह्यम् एतन् मृगेक्षणे / किम् एभिर् अस्थिरैर् देवि स्थिरम् परम् इदं धनम् / प्राणा अपि प्रदातव्या न देयं परमामृतम्",  # 165
    "देवदेव माहदेव परितृप्तास्मि शङ्कर / रुद्रयामलतन्त्रस्य सारम् अद्यावधारितम्",  # 166
    "सर्वशक्तिप्रभेदानां हृदयं ज्ञातम् अद्य च / इत्य् उक्त्वानन्दिता देवि क।ङ्थे लग्ना शिवस्य तु",  # 167
]

# English translations corresponding to VBT_CORPUS
VBT_TRANSLATIONS = [
    "O deva, I have heard the whole rudrayāmalasambhava, the trika divisions without remainder, the essence and non-essence by distinction.",
    "Even now, I have no withdrawal due to my doubts. What true form oh deva, is śabdarāśi composed of atomicities?",
    "Or what is bhairava's form in bhairavi through the nine-fold-division? Or what is the three-head-division difference? Or what is three natured śakti?",
    "Or sound and bindu composed, or indeed, what is the half-moon and impeder? [what is] the cakra acension,  or the vowel-less? Or what is the  śakti's self-form?",
    "Or parāparā's and aparṝa's sakala? Or again, if from parā, with that [parā], the parā-ness would be contradicted?",
    "Or [the form] would not certainly [be known] by difference of caste or body. Nor by the sakala from its para-ness (transcedentality) through niṣkala.",
    "O Lord, bestow your grace upon me; completely dispel all my doubts.",
    "Bhairava said:",
    "Well done, well done! You have inquired, my dear, about the very essence of the scriptures.",
    "O dear one, though it is most secret, I shall tell you; whatever little sakala forms, are bhairava's proclamation.",
    "O Goddess, the vanity is to be know. Like indra's net, māyā, dream, and indeed, the illusion of the Gandharvas city.",
    "Only for those of confused intellect, engrossed only in ostentatious rituals, [for those] minds attached to conceptualization, the meaning of meditation (dhyāna) is described.",
    "In truth, that is not navātma. Bhairava is not sound clusters. And [the true form] is not that three-headed god, nor the triad-śakti.",
    "Or [he is] neither composed of sound or bindu. Nor the half moon, nirodhika; nor broken into chakra sequence. The self-form is nor śakti.",
    "For those whose minds are not awakened, these force and fear indeed are like a mother’s sweet candies. All sacrificial affairs have been stated.",
    "Unbound by the cognitive delimitation, marking, or reckoning of space and time. Undistinguished positioning. Impossible to define. Truly inexpressible.",
    "Internal self direct experiential bliss, free from the conceptualization domain; that state full of bhairavī forms. [This is] the nature of bhairava.",
    "That form must be truly known, pure and universe pervading. In suchlike, in the transcendental category, who is to be worshiped, and who pleases?",
    "Suchlike, that bhairava's state is praised by all; She, through the parāpara form, [or] the para goddess [form], is praised.",
    "Just as the power and the possessor of power always abide [as] non-different, therefore, [so is] that quality and quality possessor. The supreme śakti is of supreme self.",
    "The burning power of fire is not distinct from śakti; Only this knowledge of reality is the beginning at entering.",
    "Having entered the śakti state, through no-difference creative contemplation; at that time, he would be in the form of śiva. Here śaivi's words are said.",
    "Just as by the light of a lamp or by the rays of the sun, the directions and their divisions are perceived in the world, so too, O beloved, is Shiva known by his power.",
    "O God of gods, marked with the trident, adorned with a skull as an ornament, you are beyond direction, space, and time, and free from all description.",
    "O Goddess, by what means does the state—full and complete—of Bhairava become manifest? How may his face indeed become Para (the supreme)? Tell me, O Bhairava, in such a way that I may know truly and completely.",
    "Upwards moves the breath, downwards the living soul; their movement brings forth creation and dissolution. In these two places of origin, sustenance arises from their filling, which is existence.",
    "Whether the winds turn within or without, like the twin movements of space, O Bhairavi, in this way the form of Bhairava is revealed by the Bhairavi’s power.",
    "When the power that is of the nature of air neither moves outward nor inward, but remains expanded; in that state of perfect non-duality in the center, there arises the true form of Bhairava.",
    "When the breath is restrained, exhaled, or filled, then at the end, that which is called ‘peace’ shines forth as peace through the power (of practice).",
    "Contemplate that subtle essence, finer than the finest, which shines like a ray from the very root; as it subsides at the end of the two sixes, there arises the awakening of Bhairava.",
    "As the lightning-like form rises, countering circle after circle, upward through three fists’ height—until, at the end, a great radiance arises.",
    "The orderly twelvefold division, rightly distinguished by the twelve-syllabled mantra, is transcended successively from gross to subtle and beyond; and ultimately, within, remains Shiva, the Liberated One.",
    "Filling the crown of the head swiftly with it, breaking through by the bridge of the eyebrow’s arch, making the mind free of all distinctions, one rises upward, ascending to all heights.",
    "As one contemplates the fivefold void adorned with the variegated forms of peacock feathers and circles, entry into the supreme void arises within the heart.",
    "In just this manner, wherever thought may arise—   On emptiness, on a wall, on another’s vessel—   It dissolves by itself, bestowing supreme blessing.",
    "Placing the mind between the eyebrows and sitting with eyes closed, gradually, through firmness of mind, one should perceive the highest goal.",
    "The central channel, established at the center, in the form of a thread of the lotus stalk— by meditating on that Goddess as the inner ether, through her the Divine shines forth.",
    "With the eyes restrained by the hand, the brows drawn together, and the passage (of breath) blocked at the gate, when the point (bindu) is seen and gradually merged into, in its midst arises the supreme state.",
    "Meditating upon the subtle spark-like point, born of the inner agitation of the radiance, shaped like a sesame seed—this point at the tip of the crest, within the heart, at the point of dissolution—one attains dissolution.",
    "One who is immersed in the soundless sound, who hears the unbroken current like a flowing river within the inner ear, who is steadfast in the sound-brahman, attains the supreme Brahman.",
    "From the utterance of the primal sound until the end of the prolonged tone, because of the experience of emptiness—   By that supreme power which is emptiness itself, O Bhairavi, one attains the state of emptiness.",
    "Whoever contemplates the beginning and end of any sound as void, he becomes empty, and his form becomes as the void itself.",
    "When, amidst the sustained and successive sounds of stringed and other musical instruments, one’s mind remains unwaveringly focused upon the end of each sound, one attains the infinite expanse of the supreme ether.",
    "Of the entire Piṅda Mantra, when pronounced in its gross sequence of syllables, and with the culmination of crescent, dot, and sound, through the intonation of the void, Śiva arises.",
    "One should contemplate the vast expanse of the sky simultaneously in all directions within one’s own body; for the one whose mind becomes thought-free, the infinite sky pervades entirely.",
    "Who contemplates, at once, the back as void and the root as void, with the power that is indifferent to the body—his mind becomes void.",
    "Contemplate firmly on the state where the back is void, the root is void, and the heart is void;   At once, due to the absence of alternatives, arises the dawn of non-duality.",
    "For a moment, let one contemplate emptiness in the body’s domain;   Being thought-free, one becomes thought-free and attains the very essence of thought-free being.",
    "All the substance within the body, O doe-eyed one, is pervaded by space; contemplating thus, steadfast becomes his meditation.",
    "In another body, consider the divisions of the skin as merely a wall;   Meditating on nothing within it, one becomes that which is beyond meditation.",
    "With eyes turned inward in the heart’s vast sky, abiding at the center of the lotus-cup, one whose mind knows no other, O lovely one, attains supreme fortune.",
    "When, with unwavering resolve, one’s mind comes to rest within the twelve-finger-space beyond all parts of one’s own body, then the true essence—one’s aim of reality—steadily reveals itself.",
    "However, wherever, at every twelfth-ending (dvādashānta), let one fix the mind;   Moment by moment, as thoughts diminish, within days the sense of separateness will vanish.",
    "When, in the end, one contemplates their own city burnt by the fire of time, arisen from the steps of time, then the appearance of peace arises.",
    "Just so, contemplating the entire world as burnt up in every way by imagination, a man of unwavering mind attains the supreme state of manhood.",
    "Whether in one’s own body or in the universe, and also the most subtle of subtle principles—whichever one meditates upon as the abode, in the end, the Supreme is revealed.",
    "Meditating on one’s frail or robust power within the twelve-fold path, entering the heart with awareness, one contemplating thus attains liberation and freedom.",
    "Meditate gradually upon all creation as the path of the worlds unfolds— from gross to subtle to the transcendental— until, at last, the mind dissolves.",
    "He who meditates upon the Shaiva principle—the truth underlying all this universe, in every direction and at every limit, through the process of the path—attains the highest greatness.",
    "O Great Goddess, consider this whole universe as empty; and having merged the mind in that (emptiness), one becomes the abode of the dissolution (of all).",
    "Casting aside awareness of the pot and the like, let one fix one's gaze upon the wall; at that very instant, merging into that (wall), by merging thus, one becomes one with that.",
    "One should fix one’s gaze on a place such as a treeless hill or wall;   When the mind becomes absorbed in that state, a person whose mental activities have dwindled arises.",
    "Meditating on the essence of both states, one should abide in the center;   By simultaneously letting go of both, the true reality shines forth in the middle.",
    "When the thought is given up and the mind restrained, it should not turn to any other object;   Then, by sustaining awareness in that very stillness, contemplation flourishes exceedingly.",
    "One should contemplate the entire body, indeed the whole universe, as pervaded by pure consciousness; simultaneously, with an undivided mind—such is the supreme awakening.",
    "From the union of the two breaths, whether within or without or ultimately inside,   the yogi becomes a vessel overflowing with the knowledge of unity.",
    "One should remember the entire world or one’s own body as filled with one’s own bliss; at once, through one’s own nectar, one becomes wholly composed of supreme bliss.",
    "By a subtle application, instantly, in the doe-eyed one,   Great joy arises—by which the truth is revealed.",
    "By gently binding all the channels, with the life-force rising upward,   At the moment when a tingling sensation is felt, supreme bliss unfolds.",
    "If, amidst fire or poison, one places the mind in happiness, or fills it solely with air, then one is united with the bliss of remembrance.",
    "The joy that arises from the union and culmination of energies, from the stirring and ultimate immersion in power—   that is the bliss of the Brahman principle; such bliss is called one’s own true bliss.",
    "O Divine Lady, even by suckling, churning, and a hundred other acts, the remembrance of a woman’s pleasure brings overwhelming bliss, even in the absence of actual power.",
    "When great joy is attained, or a beloved friend is seen after a long time, contemplate the surge of bliss that arises, merge into it, and let your mind become one with that.",
    "From the exhilaration arising from eating, drinking, and joyful delight in pleasures, one should contemplate a state of fullness; from this, great bliss will then arise.",
    "For a yogin whose soul delights solely in the unique bliss of tasting the essence of the Gita, identification with that—that is, becoming one with it—arises when his mind is wholly immersed in it.",
    "Wherever the mind finds contentment, there let the mind be sustained; for in every such place, the supreme bliss manifests itself.",
    "When future sleep has not yet come and outer objects have faded away, in that state attainable by the mind, the supreme Goddess shines forth.",
    "When the sky is variegated by the brilliance of the sun, a lamp, or other lights, and one’s gaze is fixed there, then in that very place, the true nature of one’s own Self shines forth.",
    "When, at the moment of vision, by the one who is sportive, wrathful, terrifying, and licking, and who moves through the sky, the supreme attainment is revealed.",
    "Seated on a soft seat, with buttocks as the sole support, hands and feet placed without reliance, thus, through such practice, the mind becomes supreme and fully absorbed.",
    "Sitting properly upon a seat, with both arms slightly bent, making the mind dwell in the space of the armpit—through the stillness that arises from this absorption, tranquility is attained.",
    "Fixing the gaze steadily upon the gross form of an object, and soon making the mind free from any support, one attains Shiva.",
    "Placing awareness in the center of the tongue, within the open mouth, performing utterance mentally—then, in that peace, one dissolves.",
    "Sitting, lying, or standing, meditating upon the formless—the self—within the mind, and eliminating attachment to the body, one swiftly becomes free of desires.",
    "O Devi, whether from motion while seated, or slowly by the movement of the body, when the mind becomes tranquil, one attains the divine flow.",
    "Gazing unceasingly at the clear sky, with the mind stilled, O Goddess, in that very moment one attains the form of Bhairava.",
    "Let one contemplate that all which dissolves in the crown of the head into the sky becomes Bhairava; then, all of it, having taken the form of Bhairava’s radiant essence, is absorbed into that fundamental reality.",
    "A little is known, giver of duality, by the external world—that is darkness again;   But knowing the form of Bhairava, the source of the universe and more, one becomes filled with infinite radiance.",
    "Thus, in such a difficult night, as the dark fortnight approaches and endures, contemplating the form of darkness, he shall attain the form of the terrifying Bhairava.",
    "In this way, first closing the eyes and visualizing before oneself a form as dark as Krishna, one should expand and contemplate the fierce Bhairava; thus, one becomes one with that (divinity).",
    "When, due to restraint or disruption of any one of the senses, one enters into the nondual void, there the Self alone shines forth.",
    "O Goddess, by reciting the syllable ‘A’—which is without the dot (bindu) and without the breath (visarga)—there swiftly arises a great flood of supreme knowledge.",
    "Contemplate the visarga (the final breath) at the end of a syllable with mindful attention;   With a mind freed from all supports, thus touch the eternal Brahman.",
    "One should meditate upon one's own Self as vast as the sky, unobstructed in all directions; then, the power which is consciousness, independent of all supports, will reveal its true nature.",
    "Piercing a part of the body at first with a sharp needle or the like, and then fixing one’s awareness on that very spot— in Bhairava, a pure state is attained.",
    "There is no inner action of the mind that is truly mine—thinking so, by the absence of all imaginations, one becomes freed from imaginings.",
    "The art of illusion, named Māyā, dwells as a mode of all things;   Recognizing thus the true nature of realities, one is not caught in separation.",
    "Seeing the arising desire to quarrel, one should lead it to peace;   From whatever it has arisen, into that itself let it dissolve.",
    "When neither desire nor knowledge arises in me, then—who am I, truly? Knowing this in essence, one becomes merged in that, absorbed in that reality.",
    "When desire or knowledge arises, focus the mind; with unwavering awareness of the Self, thereupon comes the realization of the true nature of reality.",
    "Knowledge that arises without cause, without foundation, is but an illusion; in truth, this does not happen to anyone thus, O beloved, such is not the nature of Shiva.",
    "Consciousness is the true nature within all bodies; there is no distinction anywhere. Therefore, recognizing all as pervaded by That, a wise person becomes free from worldly bondage.",
    "When the mind is made still amidst the spheres of desire, anger, greed, delusion, pride, and envy, then the true essence alone remains.",
    "This universe is an illusion, like a magic show or a wondrous painting— whether scattered or in motion. Yet, for one who contemplates or perceives it as such, supreme happiness arises.",
    "One should not immerse the mind in sorrow, nor let it be cast away in happiness;   O Bhairavi, know in the middle—what reality remains?",
    "Abandoning attachment to one's own body, contemplating \"I am present everywhere,\" with steadfast mind and unwavering vision—not looking elsewhere—one becomes truly happy.",
    "The knowledge that arises in me regarding a pot or the like, or desires and so on, is not considered to be all-pervading, unless one contemplates that the knower is indeed all-pervading.",
    "The awareness of the object, the subject, and their connection is common to all embodied beings; but for yogis, there is a special distinction—attentiveness to their relationship.",
    "One should perceive consciousness present in another’s body as in one’s own;   Abandoning attachment to one’s own body, one becomes all-pervading within days.",
    "Having made the mind unsupported, let one not entertain any thoughts; in that identity of self and supreme self, O gazelle-eyed one, is the state of Bhairava.",
    "He who knows all, does all, all-pervading, the Supreme Lord—   He indeed am I, steadfast in Shaiva dharma; thus, by firmness, one becomes Shiva.",
    "Just as the waves belong to water, flickerings to fire, and rays to the sun, so too are these manifold manifestations of the universe nothing but my own forms, divided as Bhairava.",
    "After whirling and whirling swiftly with the body, upon falling to the ground, with the cessation of the force of disturbance, a higher state arises.",
    "Whether through the supports, or by incapacity, or through ignorance, or by the dissolution of the mind— at the end of disturbance, when innate power enters, there shines the form of Bhairava.",
    "O goddess, listen attentively as I properly speak of this tradition; liberation arises instantly when both eyes are held perfectly still.",
    "Drawing in the senses at the ears and likewise at the lower passage, meditating on that stainless, pure, and unwavering state, let one enter the eternal Brahman.",
    "Standing above a deep pit like a well and gazing down without any wavering of thought, there arises at once a clear and complete dissolution of the mind.",
    "Wherever the mind wanders, whether outwardly or inwardly, there too is the state of Shiva; being all-pervading, where could He not be?",
    "Wherever, along the path of the senses, the consciousness of the Supreme is manifested,   Because that arises from the very nature of the elements, when awareness is dissolved, fullness of self remains.",
    "At the end of hunger and such, in fear, in sorrow, in a cave or fleeing from battle,   In wonder or at the end of curiosity and hunger, there dawns the state filled with the essence of Brahman.",
    "Among remembered objects, and in places seen, let the mind be relinquished; making one’s own body supportless, the sovereign self expands forth.",
    "At times, one should gently fix the gaze upon an object, then gradually withdraw it; O Devi, the knowledge thus united with consciousness becomes the abode of emptiness.",
    "The state of mind that arises in one who is detached and whose devotion overflows—   That is the eternal power of Shankari;   By meditating on her, thereafter, one becomes Shiva.",
    "When emptiness is perceived amidst all things, by meditating on that alone with the mind, even what is known becomes pacified.",
    "The purity that is spoken of by those with some knowledge is true purity in the vision of Shiva;   One is not pure or impure by themselves—therefore, be free of all distinctions and be happy.",
    "The presence of the Divine is everywhere, manifest even in ordinary things;   Beyond this, nothing else exists—such is the path of non-duality.",
    "He who is equal toward both enemy and friend, who remains the same in honor and dishonor—knowing that the Self is complete like Brahman—such a person becomes truly happy.",
    "Never foster hatred anywhere, nor cherish attachment at any time;   When one is free from both attachment and aversion, the Supreme Reality flows forth within.",
    "That which is unknowable, that which is ungraspable, that which is void, that which is of the nature of non-existence—all this should be conceived as Bhairava; in the end, from that, awakening arises.",
    "Placing the mind in the outer sky—eternal, without support, void, all-pervading, and free from all concepts—one should merge into the sky-less (reality).",
    "Wherever the mind wanders, let it at that very moment abandon that movement and stand unattached; thus, one becomes calm and free from all disturbance.",
    "Because fear pervades and echoes through all, and as the all-giver permeates everything—thus, by the constant utterance of the word \"Bhairava,\" one becomes Shiva.",
    "Because of the habitual notions of “I” and “mine” with regard to this (body and world), the mind, lacking a true foundation, wanders; but by the prompting of meditation on That (the Self), it becomes tranquil.",
    "Eternal, all-pervading, independent, the Lord of all—   Contemplating the words each moment,   he who has attained his aim   sees their meanings in accordance with their true nature.",
    "This entire world stands as an unreality, like a conjurer’s magic show;   What is the truth of this magic’s illusion? With firmness, one should seek to find peace.",
    "For the self, which is unchanging, where is knowledge and where is action? External objects depend on knowledge; therefore, this world is empty (unreal).",
    "I have neither bondage nor liberation; these terrors belong only to the fearful. This is but a reflection of the mind, like the sun reflected in water.",
    "All experiences of pleasure and pain enter through the gateways of the senses; thus, one who abandons the senses rests in his own true self, abiding in inner well-being.",
    "Knowledge illuminates all; the Self illuminates everything.   Because both are of a singular nature, knowledge and the object of knowledge are understood as distinct.",
    "Mind, consciousness, energy, and self—this quartet,   When, O beloved, it is completely dissolved,   Then remains the form of Bhairava alone.",
    "In brief, a hundred teachings without waves have been spoken;   O goddess, knowing these twelve and beyond, one becomes a knower of true knowledge.",
    "Here, when united with one among these, Bhairava himself is born; by speech he performs actions, bestowing either curses or blessings.",
    "He attains agelessness and immortality, endowed with the powers beginning with aṇimā; O Goddess, he becomes dear to yoginis and lord over all unions.",
    "Though living, he is free; though acting, he is not bound.",
    "Sri Devi said:",
    "O great Lord, if this body indeed belongs to another,",
    "In this established order, who is it that chants and what indeed is the chant?   Who is it that meditates, O Great Lord, who is worshipped, and who is satisfied?",
    "For whom is the oblation offered, whose is the sacrifice, what is it, and how is it performed?",
    "Lord Bhairava said:",
    "This method, O doe-eyed one, is external—applicable only to the grosser aspects.",
    "That contemplation which, again and again, is directed toward the Supreme Reality—such is true meditation. Here, that very repetition itself becomes sound, the soul of mantra, and such is the nature of this recitation.",
    "True meditation is the unwavering mind, formless and unsupported;   meditation is not the imagining of body, eyes, face, hands, and such.",
    "Worship is not with flowers and the like; it is the firm intent made upon the attributeless vast ether of consciousness. That alone is worship—when, with devotion, one’s mind dissolves therein.",
    "When one abides in either of these approaches here, day by day arises fullness of being; in this fullness, there is deep satisfaction, a state of utter completeness.",
    "In the great void, upon the fire, all sense-objects and beings are offered together with the mind; that is the sacrifice performed with the ladle of consciousness.",
    "O Supreme Lady, here the highest sacrifice is the satisfaction marked by bliss; by destroying all sins and granting protection to all, O Parvati.",
    "The union of Rudra and Shakti—that is the supreme field of contemplation;   Otherwise, how can one worship that reality, and who indeed could be satisfied?",
    "The essence of one’s own self is pure, independent bliss and consciousness, pervading all. Immersing one’s own self in its true nature—this alone is declared to be the true bathing.",
    "By whatever offerings He is worshipped, by whatever oblations the Supreme and the non-Supreme are propitiated, and whoever is the worshipper—He alone is all these; where then is the distinction of worship?",
    "The life-breath enters and departs by its own will, in twisted forms; That great goddess, of vast essence, is both the transcendent and immanent, moving through other fields.",
    "By diligently following and abiding in this supremely blissful sacrifice, one who is possessed by that divine goddess attains the supreme Bhairava.",
    "Six hundred times by day and night, along with twenty-one thousand recitations—   This is prescribed as the japa of the Goddess, easy for the devoted, but hard for the dull-witted.",
    "Thus, O Goddess, this supreme and excellent nectar has been described; but this should never be revealed to anyone.",
    "Do not bow at the feet of a wicked, cruel, or faithless teacher; only those of unwavering mind and exalted spirit, the truly valiant, deserve such reverence.",
    "For the devotees, for the lineage of the guru, one should give without hesitation—village, kingdom, town, country, sons, wife, and family.",
    "Renouncing all these things, only this should be sought, O doe-eyed one;   What use are these fleeting things, O goddess, when this is the supreme, lasting treasure?   Even life itself may be given away, but this highest nectar must never be surrendered.",
    "O God of gods, great Lord, I am fully satisfied, O Shankara; today the essence of the Rudrayamala Tantra has been realized.",
    "Today, the heart of all the manifold powers has been known;   So saying, O Goddess, delighting, she embraced Shiva’s neck.",
]

# Cross-lingual retrieval test cases: English query → Sanskrit verse
# Format: (English query, relevant_doc_indices in VBT_CORPUS)
VBT_RETRIEVAL_TEST_CASES = [
    # Opening dialogue
    ("doubt and uncertainty about true form", [1, 6]),  # VBT 1.2, 1.6b
    ("śakti and power of bhairava", [2, 3, 19, 20]),  # VBT 1.3, 1.4, 1.18, 1.19
    ("illusion māyā dream gandharva city", [10]),  # VBT 1.9
    ("meditation for confused minds rituals", [11]),  # VBT 1.10
    ("beyond time and space inexpressible", [15]),  # VBT 1.14
    ("bliss free from conceptualization bhairavī", [16]),  # VBT 1.15
    # Breath practices (prāṇa)
    ("breath ascending descending prāṇa jīva", [25, 26, 27]),  # VBT 1.24-1.27
    ("retention exhalation inhalation kumbhaka", [28]),  # VBT 1.27
    # Visualization practices
    ("subtle fire kundalini rising chakras", [29, 30, 31]),  # VBT 1.28-1.31
    ("peacock feathers circles void meditation", [33]),  # VBT 1.32
    ("central channel nāḍī lotus fiber", [36]),  # VBT 1.35
    # Sound practices
    ("unstruck sound anāhata brahman", [39]),  # VBT 1.38
    ("praṇava oṃ utterance void", [40, 41]),  # VBT 1.39-1.40
    ("stringed instruments prolonged sound", [42]),  # VBT 1.41
    # Space/void practices
    ("sky space void all directions", [44, 45, 46, 47]),  # VBT 1.43-1.46
    ("body substance pervaded by space", [48]),  # VBT 1.47
    # Gaze practices
    ("gaze clear sky mind still", [86]),  # VBT 1.84
    ("well deep pit gazing void", [117]),  # VBT 1.115
    # Mind practices
    ("mind unsupported no thoughts vikalpa", [110]),  # VBT 1.108
    ("wherever mind wanders bhairava", [118, 131]),  # VBT 1.116, 1.129
    # Pleasure and bliss
    ("sexual union śakti bliss", [71, 72]),  # VBT 1.69, 1.70
    ("joy meeting beloved friend", [73]),  # VBT 1.71
    ("eating drinking pleasure filled", [74]),  # VBT 1.72
    # Emotions and states
    ("fear sorrow anger desire", [103, 120]),  # VBT 1.101, 1.118
    ("equal enemy friend honor dishonor", [127]),  # VBT 1.125
    # Non-dual realization
    ("I am all-pervading śiva", [111]),  # VBT 1.109
    ("waves water flames fire rays sun", [112]),  # VBT 1.110
    ("no bondage no liberation fear", [137]),  # VBT 1.135
    # Final teachings
    ("hundred twelve dharaṇās teachings", [141]),  # VBT 1.139
    ("ageless immortal yoginī beloved", [143]),  # VBT 1.141
    ("worship not flowers firm mind", [153]),  # VBT 1.147
]

# VBT Similarity pairs: semantically related verses
VBT_SIMILARITY_PAIRS = [
    # === THEMATIC PAIRS (Sanskrit-Sanskrit) ===

    # Śakti nature
    (VBT_CORPUS[2], VBT_CORPUS[19]),   # śakti forms (1.3 ↔ 1.18)
    (VBT_CORPUS[19], VBT_CORPUS[20]),  # śakti and fire (1.18 ↔ 1.19)

    # Negation of forms
    (VBT_CORPUS[12], VBT_CORPUS[13]),  # not navātma, not triśira (1.11 ↔ 1.12)
    (VBT_CORPUS[10], VBT_CORPUS[11]),  # māyā illusion (1.9 ↔ 1.10)

    # Breath/prāṇa practices
    (VBT_CORPUS[25], VBT_CORPUS[26]),  # prāṇa ascending (1.24 ↔ 1.25)
    (VBT_CORPUS[25], VBT_CORPUS[27]),  # prāṇa and middle (1.24 ↔ 1.26)
    (VBT_CORPUS[27], VBT_CORPUS[28]),  # breath retention (1.26 ↔ 1.27)
    (VBT_CORPUS[65], VBT_CORPUS[68]),  # two breaths union (1.64 ↔ 1.67)

    # Kuṇḍalinī/fire visualization
    (VBT_CORPUS[29], VBT_CORPUS[30]),  # subtle fire rising (1.28 ↔ 1.29)
    (VBT_CORPUS[30], VBT_CORPUS[31]),  # chakra progression (1.29 ↔ 1.30)
    (VBT_CORPUS[31], VBT_CORPUS[32]),  # crown filling (1.30 ↔ 1.31)

    # Void/space practices
    (VBT_CORPUS[44], VBT_CORPUS[45]),  # body as space (1.43 ↔ 1.44)
    (VBT_CORPUS[45], VBT_CORPUS[46]),  # back/root/heart void (1.44 ↔ 1.45)
    (VBT_CORPUS[46], VBT_CORPUS[47]),  # void contemplation (1.45 ↔ 1.46)
    (VBT_CORPUS[47], VBT_CORPUS[48]),  # body pervaded by space (1.46 ↔ 1.47)
    (VBT_CORPUS[59], VBT_CORPUS[62]),  # universe as void (1.58 ↔ 1.61)

    # Sound/nāda practices
    (VBT_CORPUS[39], VBT_CORPUS[40]),  # anāhata and praṇava (1.38 ↔ 1.39)
    (VBT_CORPUS[40], VBT_CORPUS[41]),  # sound and void (1.39 ↔ 1.40)
    (VBT_CORPUS[41], VBT_CORPUS[42]),  # sound practices (1.40 ↔ 1.41)
    (VBT_CORPUS[42], VBT_CORPUS[43]),  # mantra and sound (1.41 ↔ 1.42)
    (VBT_CORPUS[91], VBT_CORPUS[92]),  # A-kāra and visarga (1.90 ↔ 1.91)

    # Gaze/dṛṣṭi practices
    (VBT_CORPUS[60], VBT_CORPUS[61]),  # gaze on pot/wall (1.59 ↔ 1.60)
    (VBT_CORPUS[85], VBT_CORPUS[86]),  # gazing at clear sky (1.84 ↔ 1.85)
    (VBT_CORPUS[88], VBT_CORPUS[89]),  # darkness practices (1.87 ↔ 1.88)
    (VBT_CORPUS[77], VBT_CORPUS[85]),  # light/sky gazing (1.76 ↔ 1.84)

    # Bliss/ānanda practices
    (VBT_CORPUS[66], VBT_CORPUS[67]),  # world filled with bliss (1.65 ↔ 1.66)
    (VBT_CORPUS[70], VBT_CORPUS[71]),  # śakti union bliss (1.69 ↔ 1.70)
    (VBT_CORPUS[72], VBT_CORPUS[73]),  # joy of meeting/eating (1.71 ↔ 1.72)
    (VBT_CORPUS[73], VBT_CORPUS[74]),  # eating/music bliss (1.72 ↔ 1.73)
    (VBT_CORPUS[74], VBT_CORPUS[75]),  # contentment (1.73 ↔ 1.74)

    # Mind practices
    (VBT_CORPUS[95], VBT_CORPUS[96]),  # no inner action (1.94 ↔ 1.95)
    (VBT_CORPUS[98], VBT_CORPUS[99]),  # desire/knowledge (1.97 ↔ 1.98)
    (VBT_CORPUS[110], VBT_CORPUS[118]),  # unsupported mind (1.108 ↔ 1.116)
    (VBT_CORPUS[118], VBT_CORPUS[131]),  # wherever mind wanders (1.116 ↔ 1.129)

    # Dvādaśānta practices
    (VBT_CORPUS[51], VBT_CORPUS[52]),  # twelve-finger space (1.50 ↔ 1.51)

    # Middle/madhya practices
    (VBT_CORPUS[62], VBT_CORPUS[63]),  # abide in middle (1.61 ↔ 1.62)
    (VBT_CORPUS[104], VBT_CORPUS[128]),  # middle between pleasure/pain (1.103 ↔ 1.126)

    # Non-dual realization
    (VBT_CORPUS[101], VBT_CORPUS[102]),  # consciousness everywhere (1.100 ↔ 1.101)
    (VBT_CORPUS[111], VBT_CORPUS[112]),  # I am Śiva / waves-water (1.109 ↔ 1.110)
    (VBT_CORPUS[126], VBT_CORPUS[127]),  # equal in honor/dishonor (1.124 ↔ 1.125)
    (VBT_CORPUS[127], VBT_CORPUS[128]),  # no attachment/aversion (1.125 ↔ 1.126)
    (VBT_CORPUS[136], VBT_CORPUS[137]),  # no bondage/liberation (1.134 ↔ 1.135)

    # Worship/ritual redefinition
    (VBT_CORPUS[152], VBT_CORPUS[153]),  # true worship/pūjā (1.146 ↔ 1.147)
    (VBT_CORPUS[154], VBT_CORPUS[155]),  # true sacrifice/homa (1.148 ↔ 1.149)
    (VBT_CORPUS[156], VBT_CORPUS[157]),  # true bathing (1.150 ↔ 1.151)

    # Final teachings
    (VBT_CORPUS[141], VBT_CORPUS[142]),  # 112 dhāraṇās (1.139 ↔ 1.140)
    (VBT_CORPUS[166], VBT_CORPUS[167]),  # Devī's satisfaction (1.164 ↔ 1.165)

    # === CROSS-LINGUAL PAIRS (Sanskrit-English) ===
    (VBT_CORPUS[16], VBT_TRANSLATIONS[16]),   # bliss free from vikalpa (1.15)
    (VBT_CORPUS[22], VBT_TRANSLATIONS[22]),   # lamp/sun rays (1.21)
    (VBT_CORPUS[25], VBT_TRANSLATIONS[25]),   # prāṇa ascending (1.24)
    (VBT_CORPUS[39], VBT_TRANSLATIONS[39]),   # anāhata sound (1.38)
    (VBT_CORPUS[44], VBT_TRANSLATIONS[44]),   # body as space (1.43)
    (VBT_CORPUS[70], VBT_TRANSLATIONS[70]),   # śakti union (1.69)
    (VBT_CORPUS[85], VBT_TRANSLATIONS[85]),   # gazing at sky (1.84)
    (VBT_CORPUS[110], VBT_TRANSLATIONS[110]), # unsupported mind (1.108)
    (VBT_CORPUS[111], VBT_TRANSLATIONS[111]), # I am Śiva (1.109)
    (VBT_CORPUS[127], VBT_TRANSLATIONS[127]), # no attachment (1.125)
    (VBT_CORPUS[141], VBT_TRANSLATIONS[141]), # 112 dhāraṇās (1.139)
    (VBT_CORPUS[152], VBT_TRANSLATIONS[152]), # true worship (1.146)
]

# VBT Dissimilarity pairs: unrelated verses
VBT_DISSIMILARITY_PAIRS = [
    # === CROSS-THEMATIC PAIRS (different practice domains) ===

    # Dialogue markers vs practices
    (VBT_CORPUS[7], VBT_CORPUS[25]),   # "Bhairava said" vs prāṇa practice
    (VBT_CORPUS[7], VBT_CORPUS[70]),   # "Bhairava said" vs sexual bliss
    (VBT_CORPUS[144], VBT_CORPUS[85]), # "Devī said" vs sky gazing

    # Opening questions vs specific practices
    (VBT_CORPUS[1], VBT_CORPUS[25]),   # doubt about form vs breath
    (VBT_CORPUS[1], VBT_CORPUS[70]),   # doubt vs sexual bliss
    (VBT_CORPUS[6], VBT_CORPUS[39]),   # grace request vs sound
    (VBT_CORPUS[24], VBT_CORPUS[152]), # how to know bhairava vs true worship

    # Breath vs Sound (different sensory modalities)
    (VBT_CORPUS[25], VBT_CORPUS[39]),  # prāṇa vs anāhata
    (VBT_CORPUS[28], VBT_CORPUS[42]),  # kumbhaka vs musical instruments

    # Breath vs Gaze
    (VBT_CORPUS[25], VBT_CORPUS[85]),  # prāṇa vs sky gazing
    (VBT_CORPUS[27], VBT_CORPUS[60]),  # breath middle vs pot gazing

    # Sound vs Bliss
    (VBT_CORPUS[39], VBT_CORPUS[70]),  # anāhata vs sexual union
    (VBT_CORPUS[42], VBT_CORPUS[73]),  # instruments vs eating bliss

    # Void/Space vs Bliss
    (VBT_CORPUS[44], VBT_CORPUS[70]),  # body as space vs sexual bliss
    (VBT_CORPUS[46], VBT_CORPUS[72]),  # triple void vs joy of meeting

    # Kuṇḍalinī vs Non-dual philosophy
    (VBT_CORPUS[29], VBT_CORPUS[127]), # subtle fire vs no attachment
    (VBT_CORPUS[30], VBT_CORPUS[136]), # chakra rising vs no bondage

    # Physical posture vs Mind practices
    (VBT_CORPUS[79], VBT_CORPUS[98]),  # soft seat vs desire/knowledge
    (VBT_CORPUS[80], VBT_CORPUS[110]), # armpit space vs unsupported mind

    # Opening vs closing
    (VBT_CORPUS[0], VBT_CORPUS[167]),  # śrutam deva vs devī embracing
    (VBT_CORPUS[8], VBT_CORPUS[165]),  # sādhu sādhu vs supreme nectar

    # Illusory world vs Concrete practice
    (VBT_CORPUS[10], VBT_CORPUS[25]),  # māyā illusion vs breath
    (VBT_CORPUS[134], VBT_CORPUS[70]), # world as illusion vs sexual bliss

    # Worship redefinition vs Visualization
    (VBT_CORPUS[152], VBT_CORPUS[29]), # true pūjā vs kuṇḍalinī fire
    (VBT_CORPUS[154], VBT_CORPUS[33]), # true homa vs peacock feathers

    # === COMPLETELY UNRELATED (non-tantric text) ===

    # Modern technical text vs VBT verses
    ("cooking recipe ingredients kitchen stove", VBT_CORPUS[16]),
    ("machine learning neural network training", VBT_CORPUS[110]),
    ("stock market investment portfolio returns", VBT_CORPUS[70]),
    ("software engineering agile development sprint", VBT_CORPUS[25]),
    ("climate change carbon emissions temperature", VBT_CORPUS[44]),

    # Mundane activities vs practices
    ("buying groceries at the supermarket today", VBT_CORPUS[39]),
    ("driving car highway traffic commute work", VBT_CORPUS[85]),
    ("watching television news weather forecast", VBT_CORPUS[127]),

    # Other religious/philosophical traditions (still spiritual but different)
    ("Jesus Christ resurrection salvation gospel", VBT_CORPUS[111]),
    ("Buddha enlightenment nirvana four noble truths", VBT_CORPUS[16]),
    ("Quran Allah prophet Muhammad prayer mosque", VBT_CORPUS[152]),

    # Scientific text
    ("quantum mechanics wave function probability", VBT_CORPUS[44]),
    ("DNA genetics chromosome cell division", VBT_CORPUS[29]),
    ("photosynthesis chlorophyll plant biology", VBT_CORPUS[66]),
]
